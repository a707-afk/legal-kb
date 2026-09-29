"""Agent Harness: Coordinator-based Agent execution with full observability.

Each run follows the standard loop:
    Start → Risk pre-check → Understand → Plan → [Execute per step] → Draft → Evaluate → HITL → Finalize

Every step is written to a per-run JSONL file (data/agent_runs/) for audit replay
(see BLUEPRINT D-12 附：JSONL 落盘，不再依赖 SQLAlchemy/Postgres）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── JSON extraction helpers ──────────────────────────────────────

def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Robustly extract a JSON object from LLM output.

    Tries multiple strategies: fenced code blocks, brace matching,
    and full-text parsing.
    """
    import re
    # Strategy 1: code fence
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 2: outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from LLM output."""
    import re
    # Strategy 1: code fence
    m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 2: outermost brackets
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None

def _sig(obj: Any) -> str:
    """观测/结果指纹：循环检测用（同样的证据重复出现 = 陷入循环）。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = str(obj)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def _extract_evidence(observations: list[dict]) -> list[dict]:
    """从观测里抽出检索到的证据片段（供端到端 grounding 判分 / 可观测）。"""
    out: list[dict] = []
    for obs in observations:
        res = obs.get("result") if isinstance(obs, dict) else None
        for s in ((res or {}).get("snippets") or []):
            out.append({"text": s.get("text", ""), "chunk_id": s.get("citation_id", ""),
                        "file_name": s.get("file_name", ""), "heading": s.get("heading", "")})
    return out


# Budget defaults
DEFAULT_BUDGET = {
    "max_steps": 10,
    "max_tool_calls": 20,
    "max_latency_ms": 120_000,
    "step_timeout_seconds": 30.0,
    "max_transitions": 30,
    "max_rewrite_attempts": 2,
    "loop_detect_threshold": 2,
}


@dataclass
class AgentHarnessResult:
    """Complete result of an agent run."""
    run_id: str
    status: str  # completed | failed | terminated
    final_answer: str | None = None
    final_action: str | None = None
    human_review_required: bool = False
    total_steps: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    tool_error_count: int = 0
    permission_deny_count: int = 0
    errors: list[dict] = field(default_factory=list)
    approvals: list[dict] = field(default_factory=list)
    audit_trace: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # D5/L1：本次 run 检索到的证据片段（供端到端 grounding 判分）


async def run_agent_harness(
    *,
    objective: str,
    tenant_id: str,
    user_id: str = "anonymous",
    user_context: dict[str, Any] | None = None,
    session_id: str | None = None,
    ticket_id: str | None = None,
    budget: dict[str, Any] | None = None,
) -> AgentHarnessResult:
    """Execute an agent run with full observability.

    Args:
        objective: The task objective / user query
        tenant_id: Tenant scope
        user_id: User identifier
        user_context: User roles, scopes, department
        session_id: Optional session for multi-turn
        ticket_id: Optional ticket reference
        budget: Step/tool budget limits
    """
    from src.agent.tool_registry import get_tool_registry
    from src.db.models.agent_run import AgentRun
    from src.db.models.agent_step import AgentStep
    from src.db.engine import get_sessionmaker

    t_start = time.perf_counter()
    budget = budget or DEFAULT_BUDGET
    max_steps = budget.get("max_steps", 10)
    max_tool_calls = budget.get("max_tool_calls", 20)
    max_latency_ms = budget.get("max_latency_ms", 120_000)
    step_timeout = budget.get("step_timeout_seconds", 30.0)
    max_transitions = budget.get("max_transitions", 30)
    max_rewrite_attempts = budget.get("max_rewrite_attempts", 2)

    run_id = str(uuid.uuid4())
    user_context = user_context or {}

    # ── Create AgentRun record ──
    run = AgentRun(
        id=run_id,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        ticket_id=ticket_id,
        objective=objective,
        user_query=objective,
        status="running",
        risk_level=_assess_risk(objective),
        budget_json=json.dumps(budget, ensure_ascii=False),
    )

    errors: list[dict] = []
    approvals: list[dict] = []
    audit: list[dict] = []
    total_tool_calls = 0
    tool_error_count = 0
    perm_deny_count = 0
    transition_count = 0
    loop_detect_threshold = budget.get("loop_detect_threshold", 2)
    seen_obs_sigs: dict[str, int] = {}
    loop_detected = False

    steps: list[AgentStep] = []
    step_index = 0

    def _add_step(step_type: str, input_data: dict, output_data: dict, tool_name: str | None = None,
                   tool_params: dict | None = None, tool_result: dict | None = None,
                   permission: str | None = None, latency: float = 0.0, error: str | None = None) -> AgentStep:
        nonlocal step_index
        step = AgentStep(
            id=str(uuid.uuid4()),
            run_id=run_id,
            tenant_id=tenant_id,
            step_index=step_index,
            step_type=step_type,
            input_json=json.dumps(input_data, ensure_ascii=False) if input_data else None,
            output_json=json.dumps(output_data, ensure_ascii=False) if output_data else None,
            tool_name=tool_name,
            tool_params_json=json.dumps(tool_params, ensure_ascii=False) if tool_params else None,
            tool_result_json=json.dumps(tool_result, ensure_ascii=False) if tool_result else None,
            permission_check=permission,
            latency_ms=latency,
            error_message=error,
        )
        step_index += 1
        steps.append(step)
        return step

    try:
        # Step 0.5: Input injection detection
        injection_check = _input_injection_detection(objective)
        if injection_check["blocked"]:
            _add_step("plan", {"action": "input_sanitize"}, {"blocked": True, "reason": injection_check["reason"]})
            audit.append({"step": "input_sanitize", "blocked": True, "reason": injection_check["reason"]})
            draft = "您的请求因安全原因被拦截：" + injection_check["reason"]
            _add_step("evaluate", {"action": "draft"}, {"draft": draft})
            run.status = "completed"
            run.final_answer = draft
            run.final_action = "input_blocked"
            total_latency = (time.perf_counter() - t_start) * 1000
            async with get_sessionmaker()() as session:
                run.total_steps = len(steps)
                run.total_latency_ms = total_latency
                run.audit_trace_json = json.dumps(audit, ensure_ascii=False)
                session.add(run)
                for s in steps:
                    session.add(s)
                await session.commit()
            return AgentHarnessResult(run_id=run_id, status="completed", final_answer=draft, total_steps=len(steps), audit_trace=audit)
        # ── Step 1: Risk pre-check ──
        _add_step("plan", {"action": "risk_pre_check"}, {"risk_level": run.risk_level})
        audit.append({"step": "risk_pre_check", "risk_level": run.risk_level})

        # ── Step 1.5: Intent classification (pre-plan routing) ──
        intent_info = _classify_intent(objective)
        _add_step("plan", {"action": "intent_classify"}, intent_info)
        audit.append({"step": "intent_classify", **intent_info})

        # ── Step 2: Understand & Plan ──
        plan = _build_plan(objective, run.risk_level, intent_info=intent_info)
        _add_step("plan", {"action": "build_plan", "objective": objective}, {"plan": plan})
        run.plan_json = json.dumps(plan, ensure_ascii=False)
        audit.append({"step": "plan", "steps": len(plan)})

        # ── Step 3: Execute each plan step ──
        registry = get_tool_registry()
        all_observations: list[dict] = []

        for i, step_def in enumerate(plan):
            if step_index >= max_steps:
                errors.append({"step": i, "error": "max_steps_exceeded"})
                break

            # Budget enforcement: max latency check
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            if elapsed_ms > max_latency_ms:
                errors.append({"step": i, "error": "max_latency_exceeded", "elapsed_ms": elapsed_ms})
                break

            # Budget enforcement: max transitions
            transition_count += 1
            if transition_count > max_transitions:
                errors.append({"step": i, "error": "max_transitions_exceeded"})
                break

            step_type = step_def.get("type", "execute")
            tool_name = step_def.get("tool")

            if step_type == "retrieve":
                # D5：retrieve 步真正调 local_search 取证据（原来只标记 delegated_to_rag、不检索）
                if total_tool_calls >= max_tool_calls:
                    errors.append({"step": i, "error": "max_tool_calls_exceeded"})
                    break
                import asyncio
                rq = step_def.get("query") or step_def.get("description") or objective
                t0 = time.perf_counter()
                res = None
                try:
                    res = await asyncio.wait_for(
                        registry.execute("local_search", {"query": rq, "top_k": 5},
                                         user_context=user_context, tenant_id=tenant_id),
                        timeout=step_timeout,
                    )
                except Exception as e:
                    errors.append({"step": i, "tool": "local_search", "error": f"{type(e).__name__}: {e}"})
                total_tool_calls += 1
                lat = (time.perf_counter() - t0) * 1000
                if res is not None and res.success:
                    snippets = (res.data or {}).get("snippets", [])
                    sig = _sig([s.get("citation_id") for s in snippets])
                    seen_obs_sigs[sig] = seen_obs_sigs.get(sig, 0) + 1
                    all_observations.append({"tool": "local_search", "result": res.data})
                    _add_step("retrieve", {"query": rq}, {"count": len(snippets)},
                              tool_name="local_search", latency=lat)
                    audit.append({"step": "retrieve", "query": rq[:60], "count": len(snippets),
                                  "sig_repeat": seen_obs_sigs[sig]})
                    if seen_obs_sigs[sig] > loop_detect_threshold:
                        loop_detected = True
                        run.termination_reason = "loop_detected:repeated_retrieval"
                        audit.append({"step": "loop_detected", "loop_detected": True,
                                      "where": "retrieve", "sig": sig})
                        break
                else:
                    err = (res.error if res is not None else "no_result")
                    _add_step("retrieve", {"query": rq}, {"error": err},
                              tool_name="local_search", latency=lat, error=err)
                    errors.append({"step": i, "tool": "local_search", "error": err})
                continue

            if step_type == "execute" and tool_name:
                if total_tool_calls >= max_tool_calls:
                    errors.append({"step": i, "error": "max_tool_calls_exceeded"})
                    break

                tool = registry.get(tool_name)
                if tool is None:
                    _add_step("execute", {"tool": tool_name}, {}, tool_name=tool_name,
                              error=f"Unknown tool: {tool_name}")
                    tool_error_count += 1
                    audit.append({"step": "execute", "tool": tool_name, "error": "unknown_tool"})
                    continue

                # Permission gate
                from src.agent.permission_gate import check_permission
                params = _prepare_step_params(tool_name, step_def, all_observations, objective)
                perm = check_permission(tool, user_context, params, tenant_id)

                if not perm.allowed:
                    perm_deny_count += 1
                    _add_step("execute", {"tool": tool_name, "params": params}, {},
                              tool_name=tool_name, tool_params=params,
                              permission="denied" if not perm.requires_approval else "needs_approval",
                              error=perm.reason)
                    if perm.requires_approval:
                        approvals.append({"tool": tool_name, "params": params, "reason": perm.reason})
                    audit.append({"step": "execute", "tool": tool_name, "permission": "denied", "reason": perm.reason})
                    continue

                # Execute tool（失败三分类 + 退避重试，见 BLUEPRINT D-13）
                import asyncio
                max_attempts = 1 + max(0, tool.max_retries)
                attempt = 0
                while True:
                    attempt += 1
                    t0 = time.perf_counter()
                    result = None
                    exc_msg: str | None = None
                    try:
                        result = await asyncio.wait_for(
                            registry.execute(
                                tool_name, params,
                                user_context=user_context, tenant_id=tenant_id,
                            ),
                            timeout=step_timeout,
                        )
                    except asyncio.TimeoutError:
                        exc_msg = f"harness_step_timeout ({step_timeout}s)"
                    except Exception as e:
                        exc_msg = f"{type(e).__name__}: {e}"
                    latency = (time.perf_counter() - t0) * 1000
                    total_tool_calls += 1

                    succeeded = result is not None and result.success
                    err_text = exc_msg or (result.error if result is not None else "unknown_error")
                    classification = "success" if succeeded else _classify_tool_failure(result, exc_msg)

                    _add_step(
                        "execute",
                        {"tool": tool_name, "params": params, "attempt": attempt},
                        {"success": succeeded, "classification": classification,
                         "data": result.data if result is not None else None},
                        tool_name=tool_name, tool_params=params,
                        tool_result=({"success": succeeded, "data": result.data} if result is not None else None),
                        permission="allowed", latency=latency,
                        error=None if succeeded else err_text,
                    )

                    if succeeded:
                        all_observations.append({"tool": tool_name, "result": result.data})
                        audit.append({"step": "execute", "tool": tool_name, "success": True, "attempt": attempt})
                        break

                    audit.append({"step": "execute", "tool": tool_name, "error": err_text,
                                  "classification": classification, "attempt": attempt})

                    # 可重试且未超预算 → 指数退避后重调；否则终止本次工具调用
                    if (
                        classification == "retryable"
                        and attempt < max_attempts
                        and total_tool_calls < max_tool_calls
                    ):
                        backoff = min(2 ** (attempt - 1), 8) * 0.1
                        audit.append({"step": "retry", "tool": tool_name,
                                      "attempt": attempt + 1, "backoff_s": round(backoff, 2)})
                        await asyncio.sleep(backoff)
                        continue

                    tool_error_count += 1
                    errors.append({"step": i, "tool": tool_name, "error": err_text,
                                   "classification": classification})
                    if classification == "needs_human":
                        approvals.append({"tool": tool_name, "params": params,
                                          "reason": "tool_failure_needs_human"})
                    break

            # Validate observation
            _add_step("observe", {"observations": all_observations[-3:] if all_observations else []},
                      {"count": len(all_observations)})

        # ── Step 4: Draft answer ──
        draft = _generate_draft(objective, all_observations, errors)
        _add_step("evaluate", {"action": "draft"}, {"draft": draft})
        audit.append({"step": "draft", "length": len(draft)})

        # ── Step 4.5: OutputGuard check (between draft and evaluate) ──
        try:
            from src.input_sanitizer import OutputGuard
            output_check = OutputGuard.check(draft)
            if output_check.blocked:
                draft = output_check.sanitized
                audit.append({"step": "output_guard", "blocked": True, "threats": output_check.threats})
        except Exception as e:
            logger.debug("OutputGuard check failed (non-critical): %s", e)

        # ── Step 5: Evaluate (with rewrite loop) ──
        rewrite_count = 0
        eval_result = _evaluate_result(draft, all_observations, errors)
        _add_step("evaluate", {"action": "evaluate"}, eval_result)
        audit.append({"step": "evaluate", "passed": eval_result.get("passed", True)})

        # Rewrite loop（D5：真回环——带评估反馈重新检索、追加新证据，而非复用旧观测）
        import asyncio
        while (
            not eval_result.get("passed", True)
            and rewrite_count < max_rewrite_attempts
            and not loop_detected
        ):
            rewrite_count += 1
            feedback = "；".join(str(x) for x in eval_result.get("issues", []))
            augmented_query = f"{objective} {feedback}".strip()
            # ① 接回 Retriever：用带反馈的 query 重新检索
            obs_before = len(all_observations)
            new_sig = None
            if total_tool_calls < max_tool_calls:
                try:
                    res = await asyncio.wait_for(
                        registry.execute("local_search", {"query": augmented_query, "top_k": 5},
                                         user_context=user_context, tenant_id=tenant_id),
                        timeout=step_timeout,
                    )
                    total_tool_calls += 1
                    if res is not None and res.success:
                        snippets = (res.data or {}).get("snippets", [])
                        new_sig = _sig([s.get("citation_id") for s in snippets])
                        all_observations.append({"tool": "local_search", "result": res.data,
                                                 "rewrite_round": rewrite_count})
                except Exception as e:
                    errors.append({"rewrite": rewrite_count, "error": f"{type(e).__name__}: {e}"})
            _add_step("retrieve", {"action": f"rewrite_retrieve_{rewrite_count}",
                                   "query": augmented_query, "feedback": feedback},
                      {"obs_before": obs_before, "obs_after": len(all_observations)})
            # ② 循环检测：重检索没带来新证据（签名重复或无新增）→ 陷入循环，停
            no_new = len(all_observations) == obs_before
            repeated = new_sig is not None and seen_obs_sigs.get(new_sig, 0) > 0
            if new_sig is not None:
                seen_obs_sigs[new_sig] = seen_obs_sigs.get(new_sig, 0) + 1
            if no_new or repeated:
                loop_detected = True
                run.termination_reason = "loop_detected:no_new_evidence"
                audit.append({"step": f"rewrite_{rewrite_count}", "loop_detected": True,
                              "no_new": no_new, "repeated_sig": repeated})
                break
            # ③ 用新证据重新生成 + 评估（注意：生成仍以**原始问题**提问，
            #    augmented_query 只用于检索；否则会把反馈文本当成问题写进答案）
            draft = _generate_draft(objective, all_observations, errors)
            try:
                output_check = OutputGuard.check(draft)
                if output_check.blocked:
                    draft = output_check.sanitized
            except Exception:
                pass
            eval_result = _evaluate_result(draft, all_observations, errors)
            _add_step("evaluate", {"action": f"rewrite_evaluate_{rewrite_count}"}, eval_result)
            audit.append({"step": f"rewrite_{rewrite_count}", "passed": eval_result.get("passed", True),
                          "obs_count": len(all_observations)})

        # If still not passed after rewrites, flag for human review
        if not eval_result.get("passed", True) and rewrite_count >= max_rewrite_attempts:
            audit.append({"step": "rewrite_exhausted", "forced_pass": True})

        # ── Step 6: HITL check ──
        needs_human = (
            run.risk_level in ("high", "critical")
            or bool(approvals)
            or not eval_result.get("passed", True)
            or (rewrite_count >= max_rewrite_attempts and not eval_result.get("passed", True))
        )
        if needs_human:
            _add_step("approve", {"action": "request_approval"}, {"approvals": approvals})
            audit.append({"step": "hitl", "required": True, "approvals": len(approvals)})
        else:
            audit.append({"step": "hitl", "required": False})

        # ── Step 7: Finalize ──
        total_latency = (time.perf_counter() - t_start) * 1000

        async with get_sessionmaker()() as session:
            run.status = "waiting_approval" if needs_human else "completed"
            run.final_answer = draft
            run.final_action = "human_review_required" if needs_human else "completed"
            run.human_review_required = needs_human
            run.total_steps = len(steps)
            run.total_tool_calls = total_tool_calls
            run.total_latency_ms = total_latency
            run.tool_error_count = tool_error_count
            run.permission_deny_count = perm_deny_count
            run.errors_json = json.dumps(errors, ensure_ascii=False) if errors else None
            run.approvals_json = json.dumps(approvals, ensure_ascii=False) if approvals else None
            run.audit_trace_json = json.dumps(audit, ensure_ascii=False)
            session.add(run)
            for step in steps:
                session.add(step)
            await session.commit()

        return AgentHarnessResult(
            run_id=run_id,
            status=run.status,
            final_answer=draft,
            final_action=run.final_action,
            human_review_required=needs_human,
            total_steps=len(steps),
            total_tool_calls=total_tool_calls,
            total_latency_ms=total_latency,
            tool_error_count=tool_error_count,
            permission_deny_count=perm_deny_count,
            errors=errors,
            approvals=approvals,
            audit_trace=audit,
            evidence=_extract_evidence(all_observations),
        )

    except Exception as e:
        logger.exception("Agent harness failed: run_id=%s", run_id)
        total_latency = (time.perf_counter() - t_start) * 1000
        errors.append({"error": str(e), "type": type(e).__name__})

        async with get_sessionmaker()() as session:
            run.status = "failed"
            run.termination_reason = str(e)[:128]
            run.total_steps = len(steps)
            run.total_tool_calls = total_tool_calls
            run.total_latency_ms = total_latency
            run.tool_error_count = tool_error_count + 1
            run.errors_json = json.dumps(errors, ensure_ascii=False)
            run.audit_trace_json = json.dumps(audit, ensure_ascii=False)
            session.add(run)
            for step in steps:
                session.add(step)
            await session.commit()

        return AgentHarnessResult(
            run_id=run_id,
            status="failed",
            total_steps=len(steps),
            total_tool_calls=total_tool_calls,
            total_latency_ms=total_latency,
            tool_error_count=tool_error_count + 1,
            errors=errors,
            audit_trace=audit,
        )


# ── Internal helpers ───────────────────────────────────────────────

def _input_injection_detection(objective: str) -> dict:
    """Detect prompt injection / security bypass attempts."""
    low = objective.strip().lower()
    OVERRIDE_PATTERNS = [
        "ignore previous instructions", "ignore all instructions",
        "ignore all prior instructions", "ignore all previous instructions",
        "you are now a system administrator", "you are now admin",
        "override your instructions", "override system prompt",
        "you are now a different", "act as a system administrator",
        "ignore everything above", "reset your memory",
        "now you are a", "pretend you are", "disregard previous",
    ]
    for p in OVERRIDE_PATTERNS:
        if p in low:
            return {"blocked": True, "reason": "检测到系统指令覆盖攻击 (匹配: " + p + ")"}
    CROSS_SCOPE_PATTERNS = [
        "查别人", "别人的案件", "他人卷宗", "对方的账号",
        "for user", "for another", "other user", "another party", "someone else",
        "不用走审批", "跳过审批", "不需要审批", "bypass approval",
        "不用审核", "skip review", "无需确认", "越权",
    ]
    for p in CROSS_SCOPE_PATTERNS:
        if p in low:
            return {"blocked": True, "reason": "安全策略禁止越权检索他人案件/卷宗 (匹配: " + p + ")"}
    EXPLOIT_PATTERNS = [
        "' or '", "' or 1=1", "' -- ", "1=1 --", "union select",
        "drop table", "delete from", "truncate table",
    ]
    for p in EXPLOIT_PATTERNS:
        if p in low:
            return {"blocked": True, "reason": "检测到数据库注入尝试 (匹配: " + p + ")"}
    return {"blocked": False, "reason": ""}

def _assess_risk(objective: str) -> str:
    """从请求文本评估初始风险档（法律审查场景，非一代的退款/投诉电商语义）。

    - critical：对知识库/数据的破坏性或批量导出操作
    - high：可能构成“出具正式法律意见 / 代理诉讼”等需人工复核的执业行为
    - medium：涉及时效/废止判断或敏感程序（复议/申诉/举报），需谨慎核验
    - low：一般法条检索与合规咨询
    """
    low = objective.lower()
    if any(w in low for w in ["删除", "delete", "导出", "export", "注销", "批量下载", "truncate"]):
        return "critical"
    if any(w in low for w in ["出具法律意见", "起草合同", "起草诉状", "代理诉讼", "代理仲裁",
                              "判定违法", "认定犯罪", "legal opinion", "file a lawsuit"]):
        return "high"
    if any(w in low for w in ["废止", "失效", "过期", "时效", "复议", "申诉", "举报", "信访", "仲裁"]):
        return "medium"
    return "low"


# 确定性错误标记（参数/校验/未知工具/权限）——重试无意义
_NON_RETRYABLE_MARKERS = (
    "unknown tool", "no handler", "missing required parameter",
    "should be integer", "should be string", "validation", "invalid",
    "permission denied", "requires human approval",
)


def _classify_tool_failure(result: Any, exc_msg: str | None) -> str:
    """工具失败三分类（BLUEPRINT D-13）：retryable / non_retryable / needs_human。

    - 权限拒绝或需审批 → needs_human（转 HITL，不重试）
    - 参数/校验/未知工具等确定性错误 → non_retryable（快速失败，不重试）
    - 超时或其它瞬时异常 → retryable（退避后重试）
    """
    if result is not None and getattr(result, "permission_denied", False):
        return "needs_human"
    msg = (exc_msg or (getattr(result, "error", None) if result is not None else "") or "").lower()
    if any(m in msg for m in _NON_RETRYABLE_MARKERS):
        return "non_retryable"
    return "retryable"


def _classify_intent(objective: str) -> dict[str, Any]:
    """Pre-plan intent classification.

    NOTE: The legacy keyword classifier was removed during the legal-KB
    refactoring.
    A domain intent classifier is not needed for the ReAct loop
    (the LLM planner handles decomposition). This stub returns a neutral
    intent so the legacy harness path remains runnable until Phase 2
    replaces this harness with a domain-specific harness.
    """
    return {"intent": "kb_qa", "confidence": 0.30, "emotion": None, "order_hint": ""}


def _build_plan(objective: str, risk_level: str, intent_info: dict[str, Any] | None = None) -> list[dict]:
    """Build an execution plan using LLM-based planning.

    The LLM receives the available tools and the objective, then produces
    a structured plan (list of steps with tool names and params).
    Falls back to rule-based planning if LLM is unavailable.
    """
    try:
        from src.llm import chat_completion
        from src.agent.tool_registry import get_tool_registry

        registry = get_tool_registry()
        tools_desc = []
        for t in registry.list_tools():
            tools_desc.append(f"- {t.name}: {t.description} (risk: {t.risk_level.value}, side_effect: {t.side_effect.value})")
        tools_text = "\n".join(tools_desc)

        system = (
            "你是一个法律合规研究分析 Agent 规划器。根据用户的法律问题，生成一个执行计划。\n"
            "计划是一个 JSON 数组，每个步骤包含 type、tool、params。\n"
            "type 可以是 'retrieve'（检索本地法律法规知识库）或 'execute'（调用工具）。\n"
            "只使用下面列出的可用工具，不要发明不存在的工具。\n"
            "params 中的值应基于用户请求推断，不要用占位符。\n"
            "返回纯 JSON，不要包含其他文字。"
        )
        user = (
            f"风险等级: {risk_level}\n"
            f"可用工具:\n{tools_text}\n\n"
            f"用户目标: {objective}\n\n"
            "请生成执行计划 (JSON array):"
        )

        raw = chat_completion(system, user)
        # Parse JSON from LLM response (robust extraction)
        plan = _extract_json_array(raw)
        if plan:
            logger.info("LLM plan generated: %d steps for '%s'", len(plan), objective[:50])
            return plan

        logger.warning("LLM plan parse failed, falling back to rule-based")
    except Exception as e:
        logger.warning("LLM planning failed, falling back: %s", e)

    # Fallback: rule-based planning
    return _build_plan_rule_based(objective, risk_level)


def _build_plan_rule_based(objective: str, risk_level: str) -> list[dict]:
    """Rule-based fallback planner.

    Legacy composite tools were removed during the legal-KB refactoring.
    The fallback now delegates to the local knowledge base via a
    retrieve step. Phase 2 will replace this with a ReAct loop that calls
    local_search/synthesize tools dynamically.
    """
    return [{"type": "retrieve", "description": f"Search local knowledge base for: {objective[:80]}"}]


_STATUS_LABEL = {3: "现行有效", 2: "已修订", 4: "尚未生效", 1: "已废止"}


def _currency_tag(snippet: dict) -> str:
    """把片段的时效元数据渲染成一行给 LLM 看的标注。

    为什么必须有：Agent 的核心能力是"核时效"——判"现行有效 / 尚未生效 / 未标注"。
    这些结论**只能**来自片段的 status_code / status_label / effective_date；
    证据块不带这三个字段时，LLM 只能凭法名和正文猜时效，
    实测 E 类（尚未生效）2/2 全错、F 类（未标注）一半错。
    """
    sc = snippet.get("status_code")
    try:
        sc = int(sc) if sc is not None else None
    except (TypeError, ValueError):
        sc = None
    if sc is None:
        return "时效: 未标注（无法确定是否现行有效）"
    label = str(snippet.get("status_label") or "").strip() or _STATUS_LABEL.get(sc, "")
    out = f"时效: {label or '未知'}(status_code={sc})"
    eff = str(snippet.get("effective_date") or "").strip()
    if eff:
        out += f"，生效日期 {eff}"
    return out


def _snippet_head(snippet: dict, *, with_score: bool = True) -> str:
    """拼片段抬头：法名 / 章条 / 时效（/ 分数）。"""
    parts = [
        str(snippet.get("title") or "").strip(),
        str(snippet.get("file_name") or "").strip(),
        str(snippet.get("heading") or "").strip(),
        _currency_tag(snippet),
    ]
    head = " / ".join(x for x in parts if x)
    score = snippet.get("score")
    if with_score and score is not None:
        head += f" (score={score})"
    return head


def _format_evidence_block(observations: list[dict], *, per_snippet_chars: int = 900,
                          total_chars: int = 8000) -> str:
    """把观测里的检索片段拼成**给 LLM 看的编号证据块**。

    为什么不能 `json.dumps(result)[:500]`：local_search 的 result 是
    `{"query","retrieval_query","count","snippets":[...]}`，500 字符连第一条
    法条原文都读不到——LLM 只能凭标题猜，产出无法溯源的答案，随后 grounding
    判分必挂、回环空转（实测 D 类 15 题 grounding_rate=0.0）。
    这里直接摊平 snippets，保留 文件/标题/条号/原文，并给每条编号，便于
    生成侧写 [1][2] 引用、评估侧逐条核对。
    """
    lines: list[str] = []
    used = 0
    n = 0
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        tool = obs.get("tool", "unknown")
        result = obs.get("result")
        if not isinstance(result, dict):
            continue
        snippets = result.get("snippets") or []
        for s in snippets:
            if not isinstance(s, dict):
                continue
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            n += 1
            block = f"[{n}] {_snippet_head(s)}\n{text[:per_snippet_chars]}"
            if used + len(block) > total_chars:
                lines.append(f"...（证据过长，已截断，共 {len(snippets)} 条）")
                break
            lines.append(block)
            used += len(block)
        # synthesize 等工具直接给答案文本
        ans = result.get("answer")
        if isinstance(ans, str) and ans.strip():
            lines.append(f"[{tool} 综合稿] {ans.strip()[:per_snippet_chars]}")
        if used > total_chars:
            break
    return "\n\n".join(lines) if lines else "（本次未检索到任何片段）"


def _collect_evidence_texts(observations: list[dict], *, limit: int = 8) -> list[str]:
    """从观测里取出片段原文（带时效抬头），供 synthesize 等工具在**执行期**使用。"""
    texts: list[str] = []
    for obs in observations:
        res = obs.get("result") if isinstance(obs, dict) else None
        for s in ((res or {}).get("snippets") or []):
            t = str(s.get("text") or "").strip()
            if t:
                texts.append(f"{_snippet_head(s, with_score=False)}\n{t}")
    return texts[:limit]


def _prepare_step_params(tool_name: str, step_def: dict, observations: list[dict],
                         objective: str) -> dict:
    """补齐规划器**在计划期无法提供**的运行时参数。

    规划器只看得到工具签名、看不到检索结果，所以 `synthesize` 的 `evidence`
    它只能瞎填（实测：填成一句说明字符串 "基于检索结果综合回答…"）。工具校验
    随即失败 → 记入 errors → 评估判失败 → 回环重检索拿回同样证据 → 判循环
    → 好答案被推进人工审核（实测 D-0123 就是这条路径）。
    这里在执行期把已检索到的片段原文注入，让 synthesize 真正可用。
    """
    params = dict(step_def.get("params") or {})
    if tool_name != "synthesize":
        return params
    ev = params.get("evidence")
    if not isinstance(ev, list) or not ev:
        texts = _collect_evidence_texts(observations)
        if texts:
            params["evidence"] = texts
    if not str(params.get("question") or "").strip():
        params["question"] = objective
    return params


def _generate_draft(objective: str, observations: list[dict], errors: list[dict]) -> str:
    """Generate a draft answer using LLM with tool observations as context."""
    context = _format_evidence_block(observations)
    error_text = json.dumps(errors[-5:], ensure_ascii=False)[:300] if errors else "无错误"

    try:
        from src.llm import chat_completion
        system = (
            "你是一个法律合规研究分析 Agent。根据用户法律问题和检索到的资料，生成一个专业、客观的分析回复。\n"
            "规则：\n"
            "1. 只基于提供的证据片段回答，不要编造法条、条款号或生效日期；无法在证据中找到依据的内容一律不写\n"
            "2. 关键事实陈述后标注证据编号，如 [1][2]，编号必须对应下方证据块\n"
            "3. 引用法条时直接摘录证据中的原文，不要改写条号与款项\n"
            "4. 若证据不足以回答（例如问题超出知识库范围、或未检索到相关法条），"
            "必须明确说明「未检索到相关内容，超出本知识库范围」，不要用常识或推测补齐\n"
            "5. 证据块每条抬头都带时效标注（现行有效/已修订/尚未生效/已废止/未标注 + 生效日期），"
            "涉及法条时效或版本时**必须依据该标注**作答：\n"
            "   - 标注「尚未生效」→ 明确说明「该版本尚未生效」，并给出标注里的生效日期；\n"
            "   - 标注「未标注」→ 明确说明「时效状态未标注、无法确定是否现行有效」，不得断言现行有效；\n"
            "   - 标注「已废止/已修订」→ 说明已被取代，并优先引用现行有效版本。\n"
            "6. 回复要简洁、专业、客观，并注明仅供参考、不构成法律意见\n"
            "7. 不要暴露内部系统名称或技术细节\n"
            "8. 不要输出与证据无关的章节标题、目录或表格装饰"
        )
        user = (
            f"用户问题: {objective}\n\n"
            f"证据片段:\n{context}\n\n"
            f"执行过程中的错误: {error_text}\n\n"
            "请生成回复（关键结论后标注 [n] 证据编号）:"
        )
        draft = chat_completion(system, user)
        if draft and len(draft) > 10:
            logger.info("LLM draft generated: %d chars", len(draft))
            return draft
    except Exception as e:
        logger.warning("LLM draft generation failed: %s", e)

    # Fallback: template-based draft
    if not observations:
        return f"针对研究问题「{objective}」，我暂时未检索到足够资料，建议换个角度检索或扩大范围。"
    parts = []
    for obs in observations[-5:]:
        result = obs.get("result", {})
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, str) and len(v) < 200:
                    parts.append(f"{k}: {v}")
    if parts:
        return f"检索结果（{objective}）：\n" + "\n".join(f"  - {p}" for p in parts[:10])
    return f"已就「{objective}」检索相关资料，但证据不足，建议补充检索或转人工复核。"


def _evaluate_result(draft: str, observations: list[dict], errors: list[dict]) -> dict:
    """Evaluate the result using LLM-based grounding verification.

    Checks:
    1. Does the draft reference the observations?
    2. Are there unsupported claims?
    3. Is the response safe (no PII, no injection)?
    """
    blocking: list[str] = []
    warnings: list[str] = []

    # Basic structural checks
    if len(draft) < 10:
        blocking.append("draft_too_short")
    if not observations:
        blocking.append("no_observations")
    if len(errors) > 0:
        # 工具报错只是质量信号，**不判失败**：回环只做「带反馈重检索」，
        # 修不了工具参数类错误；拿它判失败只会空转一轮再把好答案推人工。
        warnings.append(f"{len(errors)}_tool_errors")

    # LLM-based grounding check
    try:
        from src.llm import chat_completion
        # ⚠️ 必须把**证据原文**给审查员，而不是键名。
        # 旧代码只传 `result_keys`（["query","count","snippets"]），审查员看不到任何
        # 法条文本，只能一律判 grounded=false → 每次评估必挂 → 触发 rewrite 回环 →
        # 重检索签名重复 → loop_detected → 全量转人工。实测 15/15 题落 HITL。
        obs_text = _format_evidence_block(observations, per_snippet_chars=500, total_chars=4000)

        system = (
            "你是一个研究回复质量审查员。检查 Agent 的回复是否基于检索到的证据片段。\n"
            "返回 JSON: {\"grounded\": true/false, \"unsupported_claims\": [...], \"safe\": true/false}\n"
            "grounded=true 表示回复中的事实陈述都能在证据片段里找到依据"
            "（含对证据原文的摘录、转述与条号引用）；\n"
            "回复中若明确声明『未检索到/超出知识库范围/时效未标注』属于诚实表述，不算 unsupported。\n"
            "仅当回复写出了证据中不存在的事实（编造条号、生效日期、条文内容）时判 grounded=false，"
            "并把这类句子放进 unsupported_claims。\n"
            "safe=true 表示没有暴露内部系统名、API key、个人信息。\n"
            "只返回 JSON，不要其他文字。"
        )
        user = (
            f"Agent 回复:\n{draft[:1500]}\n\n"
            f"检索到的证据片段:\n{obs_text}\n\n"
            "请评估 (JSON):"
        )
        raw = chat_completion(system, user)
        eval_data = _extract_json_object(raw)
        if eval_data:
            grounded = eval_data.get("grounded", False)
            safe = eval_data.get("safe", True)
            unsupported = eval_data.get("unsupported_claims", [])
            if not grounded:
                blocking.append("not_grounded")
            if not safe:
                blocking.append("unsafe_content")
            if unsupported:
                warnings.append(f"unsupported: {unsupported[:3]}")
    except Exception as e:
        logger.warning("LLM evaluation failed: %s", e)

    passed = len(blocking) == 0
    return {
        "passed": passed,
        "observations_count": len(observations),
        "error_count": len(errors),
        "draft_length": len(draft),
        # issues 供回环构造反馈文本：blocking 优先，warnings 作补充
        "issues": blocking + warnings,
        "blocking_issues": blocking,
        "warnings": warnings,
    }
