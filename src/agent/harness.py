"""Agent Harness: Coordinator-based Agent execution with full observability.

Harness V2（Phase 1）：控制流从“顺序 13 步”重构为**显式状态机** while 循环 +
handler 分发（状态节点见 src/agent/state_machine.py::HarnessState）：

    INIT → SECURITY_CHECK → CONTEXT_LOAD → RISK_INTENT → ROUTE
      ├─ FAST_PATH ─────────────────────────────┐
      └─ PLAN → PLAN_VALIDATE → EXECUTE_LOOP ⇄ CHECKPOINT
                                   │
                 DRAFT → OUTPUT_GUARD → EVALUATE ⇄ REWRITE
                                   │
                 HITL_OUTPUT → PERSIST → DONE
      （异常/预算/计划非法 → ERROR_CLASSIFY → RETRY_BACKOFF / DEGRADED）

每个 handler 只做一件事并返回下一个状态；FULL_PIPELINE 路径的行为与重构前
保持一致（现有测试覆盖），FAST_PATH / 错误处理路径为新增能力，默认由 feature
flag 控制，不改变既有默认行为。

每一步仍写入 per-run JSONL 文件（data/agent_runs/）供审计回放
（见 BLUEPRINT D-12 附：JSONL 落盘，不再依赖 SQLAlchemy/Postgres）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agent.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    HISTORY_SUMMARY_SECTION,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
)
from src.agent.state_machine import (
    BUDGET_FAST,
    BUDGET_FULL,
    BUDGET_HITL,
    BudgetTracker,
    Checkpoint,
    ErrorAction,
    HarnessState,
    RouteDecision,
    TERMINAL_STATES,
)

logger = logging.getLogger(__name__)


# ── Plan schema (pydantic) ────────────────────────────────────────────────────
#
# 为什么要有这个模型：旧代码靠 ``_extract_json_array`` 正则+括号硬解析 LLM 的纯文本
# 输出，任何格式漂移（多一个字段、少一个引号、多一段解释文字）都会让整份计划被
# 丢弃、静默回退到规则兜底。用 pydantic 做强类型校验后：① 关键字段缺失/类型错
# 立刻可见 ② 未知字段通过 ``extra="allow"`` 保留，兼容 LLM 偶尔多输出的辅助键。

class PlanStep(BaseModel):
    """执行计划中的一步。

    - ``type="retrieve"``：走 local_search 检索证据，``query``/``description`` 至少一项非空
    - ``type="execute"``：调用注册表里的具名工具，``tool`` 必填
    """
    model_config = ConfigDict(extra="allow")

    type: Literal["retrieve", "execute"] = "execute"
    tool: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    query: str | None = None
    description: str | None = None


# OpenAI function-calling schema：把「生成计划」本身建模成一个工具，
# 让后端用原生 tool_calls 通道返回结构化参数，避开自由文本解析。
PLAN_TOOL_NAME = "build_plan"
PLAN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": PLAN_TOOL_NAME,
        "description": (
            "根据用户法律问题构建一个执行计划。计划是步骤数组，每步包含 type、tool、"
            "params。type 为 'retrieve' 表示检索本地法律法规知识库，'execute' 表示调用工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "执行计划的步骤数组，按执行顺序排列",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["retrieve", "execute"],
                                "description": "步骤类型：retrieve=检索知识库，execute=调用工具",
                            },
                            "tool": {
                                "type": "string",
                                "description": "工具名（type=execute 时必填，且必须是可用工具之一）",
                            },
                            "params": {
                                "type": "object",
                                "description": "工具参数键值对",
                                "additionalProperties": True,
                            },
                            "query": {
                                "type": "string",
                                "description": "检索查询词（type=retrieve 时使用，优先用法律术语）",
                            },
                            "description": {
                                "type": "string",
                                "description": "步骤的人类可读描述",
                            },
                        },
                        "required": ["type"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
}

# 强制 LLM 调用 build_plan（避免它回退到自由文本）
PLAN_TOOL_CHOICE: dict[str, Any] = {
    "type": "function",
    "function": {"name": PLAN_TOOL_NAME},
}


def _validate_plan(raw: Any) -> tuple[list[dict] | None, list[str]]:
    """用 pydantic 校验计划数组。

    Returns:
        ``(plan, errors)``：全部步骤通过校验时返回 ``(list[dict], [])``；
        任一步骤不合法时返回 ``(None, [错误描述, ...])``。调用方据此决定
        是否回退到 ``_build_plan_rule_based``。
    """
    if not isinstance(raw, list) or not raw:
        return None, ["plan must be a non-empty list"]

    errors: list[str] = []
    validated: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"step[{i}]: not a dict (got {type(item).__name__})")
            continue
        try:
            step = PlanStep.model_validate(item)
        except ValidationError as ve:
            errors.append(f"step[{i}]: {ve.errors()[:2]}")
            continue
        # 语义级校验：execute 必须带 tool；retrieve 必须带 query/description 之一
        if step.type == "execute" and not (step.tool or "").strip():
            errors.append(f"step[{i}]: type=execute 但缺少 tool")
            continue
        if step.type == "retrieve" and not (
            (step.query or "").strip() or (step.description or "").strip()
        ):
            errors.append(f"step[{i}]: type=retrieve 但 query/description 均为空")
            continue
        # 用 model_dump 保留 extra 字段（向后兼容旧计划里可能存在的自定义键）
        validated.append(step.model_dump(exclude_none=False))

    if errors:
        return None, errors
    return validated, []


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


def _get_session_summary(session_id: str | None) -> str:
    """多轮对话：取会话历史摘要（防御式，任何失败都不阻断主流程）。

    - session_id 为 None / 无历史 → 返回空串（单轮模式，不注入）
    - 摘要按 HISTORY_SUMMARY_MAX_CHARS 截断，计入 token 预算
    - summarize_last_n 是纯内存字符串操作，不调 LLM，无循环依赖/性能风险
    """
    if not session_id:
        return ""
    try:
        from src.session_mgr import get_session
        mem = get_session(session_id)
        if mem is None or mem.is_empty():
            return ""
        summary = mem.summarize_last_n(HISTORY_SUMMARY_TURNS)
        return summary[:HISTORY_SUMMARY_MAX_CHARS]
    except Exception as e:
        logger.warning("Session summary unavailable (non-critical): %s", e)
        return ""


def _record_session_turn(session_id: str | None, objective: str, answer: str) -> None:
    """把本轮问答写回会话记忆，供下一轮取摘要；失败不阻断主流程。"""
    if not session_id:
        return
    try:
        from src.session_mgr import record_turn
        record_turn(session_id, user_message=objective, assistant_message=answer)
    except Exception as e:
        logger.warning("Session record_turn failed (non-critical): %s", e)


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


# ── 模型上下文窗口配置（BLUEPRINT D-18 Context Engineering 的一部分）──────
#
# 换模型时只改这一处。默认值对应 src/llm.py 的定案模型
# sensenova-6.8-flash-lite（窗口 128000 tokens，见 docs/06 §四 D-24）。
MODEL_CONTEXT_WINDOW: int = 128_000   # 模型上下文窗口（tokens）
TOKEN_ESTIMATE_RATIO: float = 1.5     # 中文经验系数：1 字 ≈ 1.5 token
EVIDENCE_BUDGET_RATIO: float = 0.3    # 证据块最多占窗口的 30%（留给 system prompt + 用户输入 + 输出）

#: 旧版 _format_evidence_block 的硬编码值（total_chars=8000 / 评估器 4000）。
#: 换算回窗口占比：8000 字 × 1.5 token/字 = 12000 tokens = 128000 的 9.375%。
#: 默认比例沿用这个换算，保证**配置化不改变现有行为**（实测预算仍是
#: 8000/4000 字）；EVIDENCE_BUDGET_RATIO 则作为硬上限护栏生效。
_GENERATOR_EVIDENCE_BUDGET_CHARS: int = 8_000
EVIDENCE_BUDGET_RATIO_DEFAULT: float = (
    _GENERATOR_EVIDENCE_BUDGET_CHARS * TOKEN_ESTIMATE_RATIO / MODEL_CONTEXT_WINDOW
)  # = 0.09375
EVALUATOR_EVIDENCE_BUDGET_RATIO: float = EVIDENCE_BUDGET_RATIO_DEFAULT / 2  # 评估器更紧：4000 字

#: 多轮历史摘要注入预算（字符）。摘要在生成器 prompt 里只是背景，
#: 占窗口 ≤ 0.75%（≈1200 tokens），不能挤占证据预算。
HISTORY_SUMMARY_MAX_CHARS: int = 800
#: 取最近几轮对话做摘要（session_mgr.summarize_last_n 的 n）
HISTORY_SUMMARY_TURNS: int = 3


def _chars_budget(ratio: float) -> int:
    """窗口占比 → 字符预算：window × ratio ÷ (token/字)，且不超过硬上限。

    为什么钉上限：证据块之外还要留 system prompt、用户输入、历史摘要和
    输出 tokens；即使有人把默认比例调过 0.3，实际预算也不会突破
    EVIDENCE_BUDGET_RATIO（BLUEPRINT D-18：超预算时宁可裁证据）。
    """
    effective = min(ratio, EVIDENCE_BUDGET_RATIO)
    return int(MODEL_CONTEXT_WINDOW * effective / TOKEN_ESTIMATE_RATIO)


#: 默认证据块字符预算（= 8000，与配置化前的硬编码值一致）
DEFAULT_EVIDENCE_TOTAL_CHARS: int = _chars_budget(EVIDENCE_BUDGET_RATIO_DEFAULT)
#: 评估器证据块字符预算（= 4000）
EVALUATOR_EVIDENCE_TOTAL_CHARS: int = _chars_budget(EVALUATOR_EVIDENCE_BUDGET_RATIO)


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


# ── Harness V2: 状态机上下文（承载所有中间态）────────────────────────────
#
# 为什么用 dataclass 而非 dict：状态机 handler 之间靠 ctx 传递几十个字段，
# dict 的键名拼写错误只会在运行期静默产生 None；dataclass 把字段固化成属性，
# 改错名字立刻 AttributeError，且 IDE/类型检查可见。字段按关注点分组。

@dataclass
class HarnessContext:
    """一次 Agent run 的全部中间状态，在状态机 handler 之间传递。"""

    # ── 身份 / 输入 ──
    run_id: str
    objective: str
    tenant_id: str
    user_id: str
    user_context: dict[str, Any]
    session_id: str | None
    ticket_id: str | None
    t_start: float
    budget: dict[str, Any]
    run: Any                                   # src.db.models.agent_run.AgentRun

    # ── 预算（由 _apply_caller_budget / _apply_route_budget 填充）──
    max_steps: int = 10
    max_tool_calls: int = 20
    max_latency_ms: float = 120_000
    step_timeout: float = 30.0
    max_transitions: int = 30                  # 执行循环内 per-step 计数上限（沿用旧语义）
    max_rewrite_attempts: int = 2
    loop_detect_threshold: int = 2
    state_transition_cap: int = 200            # 状态机级转移护栏（独立、宽松，仅防失控打转）
    #: Phase 4：运行时预算消耗追踪器（默认按 BUDGET_FULL 初始化，RISK_INTENT 定案后
    #: 同步为实际路由档位的 RouteBudget）。与上面 max_* 字段是**并行的两层护栏**：
    #: max_* 沿用旧的 per-step 计数语义（_run_plan_step 内部检查），budget_tracker 是
    #: 状态机级的软护栏 + 审计快照来源（_handle_execute_loop / _emit_audit 使用）。
    budget_tracker: BudgetTracker = field(default_factory=lambda: BudgetTracker(budget=dict(BUDGET_FULL)))

    # ── 路由 ──
    risk_level: str = "low"
    intent_info: dict[str, Any] = field(default_factory=dict)
    route_decision: Any = None                 # state_machine.RouteDecision
    from_fast: bool = False

    # ── 上下文层 ──
    session_raw: str = ""                    # Layer 1: 最近 N 轮原文（零 LLM 成本）
    session_summary: str = ""                # Layer 2: 惰性摘要（按条件触发）
    context_layers_used: list[str] = field(default_factory=list)

    # ── 规划 / 执行 ──
    registry: Any = None
    plan: list[dict] = field(default_factory=list)
    plan_cursor: int = 0
    observations: list[dict] = field(default_factory=list)
    seen_obs_sigs: dict[str, int] = field(default_factory=dict)
    loop_detected: bool = False
    total_tool_calls: int = 0
    tool_error_count: int = 0
    perm_deny_count: int = 0
    transition_count: int = 0                  # 执行循环内 per-step 计数

    # ── 草稿 / 评估 ──
    draft: str = ""
    eval_result: dict | None = None
    rewrite_count: int = 0
    guard_blocked: bool = False                # OUTPUT_GUARD 是否拦截（→ HITL_OUTPUT）
    output_guard_entry: str = ""               # 本次进 OUTPUT_GUARD 的来源状态（审计用）

    # ── HITL ──
    approvals: list[dict] = field(default_factory=list)
    hitl_approvals: list[dict] = field(default_factory=list)
    needs_human: bool | None = None
    hitl_type: str | None = None               # pre | action_approval | output_approval
    pending_hitl: dict[str, Any] | None = None  # EXECUTE_LOOP 暂存的待审批动作
    reject_reason: str | None = None           # ROUTE/审批拒绝原因

    # ── 错误处理 ──
    errors: list[dict] = field(default_factory=list)
    error_origin_state: Any = None
    error_action: Any = None
    error_detail: dict[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0
    degraded_forced: bool = False

    # ── 审计 / 断点 ──
    steps: list[Any] = field(default_factory=list)          # AgentStep 列表（落盘用）
    step_index: int = 0
    audit: list[dict] = field(default_factory=list)         # 业务审计（→ result.audit_trace，沿用旧格式）
    state_events: list[dict] = field(default_factory=list)  # 状态转移审计（Phase 4 落盘）
    checkpoints: list[dict] = field(default_factory=list)
    _audit_meta_buffer: dict[str, Any] = field(default_factory=dict)
    #: 已经落盘的 state_events 数量（增量 flush 用，避免重复写入 JSONL）
    _audit_flushed_count: int = 0

    # ── 最终结果（由 _persist_run 填充，_build_result 读取）──
    result_status: str = "completed"
    result_final_answer: str | None = None
    result_final_action: str | None = None
    result_needs_human: bool = False
    result_latency_ms: float = 0.0

    # ── 方法 ──
    def add_step(self, step_type: str, input_data: dict, output_data: dict, *,
                 tool_name: str | None = None, tool_params: dict | None = None,
                 tool_result: dict | None = None, permission: str | None = None,
                 latency: float = 0.0, error: str | None = None) -> Any:
        """追加一条 AgentStep 记录（等价于旧 harness 里的 _add_step 闭包）。"""
        from src.db.models.agent_step import AgentStep
        step = AgentStep(
            id=str(uuid.uuid4()),
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            step_index=self.step_index,
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
        self.step_index += 1
        self.steps.append(step)
        return step

    def set_audit_meta(self, **kwargs: Any) -> None:
        """暂存本次状态转移的审计元数据（_emit_audit 取走后清空）。"""
        self._audit_meta_buffer.update(kwargs)

    def pop_audit_metadata(self) -> dict[str, Any]:
        meta = self._audit_meta_buffer
        self._audit_meta_buffer = {}
        return meta


def _apply_caller_budget(ctx: HarnessContext) -> None:
    """把调用方传入的 budget（或 DEFAULT_BUDGET）映射到 ctx 的各上限字段。

    保持与重构前完全一致的取值与默认，确保 FULL_PIPELINE 行为不变。
    """
    b = ctx.budget
    ctx.max_steps = b.get("max_steps", 10)
    ctx.max_tool_calls = b.get("max_tool_calls", 20)
    ctx.max_latency_ms = b.get("max_latency_ms", 120_000)
    ctx.step_timeout = b.get("step_timeout_seconds", 30.0)
    ctx.max_transitions = b.get("max_transitions", 30)
    ctx.max_rewrite_attempts = b.get("max_rewrite_attempts", 2)
    ctx.loop_detect_threshold = b.get("loop_detect_threshold", 2)
    ctx.state_transition_cap = b.get("max_state_transitions", 200)


def _apply_route_budget(ctx: HarnessContext, rb: dict[str, Any]) -> None:
    """用 RouteBudget（FAST/HITL 档）覆盖 ctx 上限；full 档不调用本函数（保留调用方预算）。"""
    ctx.max_steps = rb.get("max_steps", ctx.max_steps)
    ctx.max_tool_calls = rb.get("max_tool_calls", ctx.max_tool_calls)
    ctx.max_latency_ms = rb.get("max_latency_ms", ctx.max_latency_ms)
    ctx.max_rewrite_attempts = rb.get("max_rewrite_retries", ctx.max_rewrite_attempts)
    ctx.loop_detect_threshold = rb.get("loop_detect_threshold", ctx.loop_detect_threshold)


def _decide_route(ctx: HarnessContext) -> RouteDecision:
    """RISK_INTENT 阶段的路由决策。

    Phase 1 默认（feature flag 关）一律走 ``full``，与重构前 FULL_PIPELINE 行为
    完全一致——现有测试与评测都依赖这条路径。开启 ``HARNESS_FAST_PATH`` 后才启用
    分流：low+kb_qa → fast（跳过规划器，直接检索+生成）；high/critical → hitl_pre。
    这样新增能力默认不改变既有行为，可灰度验证后再放量。
    """
    risk = ctx.risk_level
    intent = ctx.intent_info.get("intent", "kb_qa")
    fast_enabled = os.getenv("HARNESS_FAST_PATH", "0").strip().lower() in ("1", "true", "yes", "on")
    if fast_enabled:
        if risk in ("high", "critical"):
            return RouteDecision(route="hitl_pre", risk_level=risk, intent=intent, budget=BUDGET_HITL)
        if risk == "low" and intent == "kb_qa":
            return RouteDecision(route="fast", risk_level=risk, intent=intent, budget=BUDGET_FAST)
    return RouteDecision(route="full", risk_level=risk, intent=intent, budget=BUDGET_FULL)


def _compute_needs_human(ctx: HarnessContext) -> bool:
    """是否需要人工复核（沿用旧 Step 6 判据；eval_result 为 None 时视为已通过）。"""
    passed = (ctx.eval_result or {}).get("passed", True)
    return (
        ctx.risk_level in ("high", "critical")
        or bool(ctx.approvals)
        or not passed
        or (ctx.rewrite_count >= ctx.max_rewrite_attempts and not passed)
    )


def _hitl_output_reason(ctx: HarnessContext) -> str:
    """归纳 HITL_OUTPUT 触发人工复核的原因（供审计/审批单展示，Phase 2, 2.3）。"""
    if ctx.guard_blocked:
        return "output_guard_blocked"
    passed = (ctx.eval_result or {}).get("passed", True)
    if ctx.risk_level in ("high", "critical"):
        return f"high_risk:{ctx.risk_level}"
    if not passed and ctx.rewrite_count >= ctx.max_rewrite_attempts:
        return "rewrite_exhausted"
    if not passed:
        return "eval_failed"
    if ctx.approvals:
        return "action_approval_required"
    return "manual_review"


def _simulate_hitl_pre_approval(ctx: HarnessContext) -> bool:
    """Phase 2 模拟事前审批决策：默认通过（真实异步审批需 WebSocket/回调，不在本期）。

    预留可注入的拒绝通道：若调用方在 user_context 里显式给出
    ``hitl_pre_approved=False``，则视为审批被拒（供测试/灰度演练拒绝路径）。
    """
    uc = ctx.user_context or {}
    decision = uc.get("hitl_pre_approved")
    if decision is None:
        return True  # Phase 2：直接视为通过
    return bool(decision)


def _sync_latency(ctx: HarnessContext) -> None:
    """把 budget_tracker.latency_ms 同步为距 run 开始的真实壁钟毫秒数。"""
    ctx.budget_tracker.latency_ms = int((time.perf_counter() - ctx.t_start) * 1000)


def _estimate_tokens(obj: Any) -> int:
    """粗估一段工具结果占用的 token 数（沿用中文经验系数 TOKEN_ESTIMATE_RATIO）。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - 不可序列化对象退化到 str()
        s = str(obj)
    return int(len(s) * TOKEN_ESTIMATE_RATIO)


def _prompt_version_hash() -> str:
    """计算所有 prompt 的版本哈希（用于 A/B 对比分组，同内容必得同哈希）。"""
    all_prompts = "|".join([
        PLANNER_SYSTEM_PROMPT,
        GENERATOR_SYSTEM_PROMPT,
        EVALUATOR_SYSTEM_PROMPT,
        SYNTHESIZE_SYSTEM_PROMPT,
    ])
    return hashlib.sha256(all_prompts.encode("utf-8")).hexdigest()[:12]


def _safe_name(run_id: str) -> str:
    """run_id → 安全文件名（与 src.trace.trace_path 的清洗规则保持一致）。"""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (run_id or "unknown"))


def _emit_audit(ctx: HarnessContext, prev_state: Any, to_state: Any) -> None:
    """记录一条状态转移审计事件（内存 list），并自动附带 budget_consumed 快照。

    Phase 4：每次转移都刷新 latency/transitions 计数，并把当前预算消耗快照
    写进 metadata（若 handler 已显式设置则不覆盖，用 setdefault）。
    """
    _sync_latency(ctx)
    ctx.budget_tracker.transitions += 1
    meta = ctx.pop_audit_metadata()
    meta.setdefault("budget_consumed", ctx.budget_tracker.snapshot())
    event = {
        "run_id": ctx.run_id,
        "from_state": getattr(prev_state, "value", str(prev_state)),
        "to_state": getattr(to_state, "value", str(to_state)),
        "timestamp": time.time(),
        "metadata": meta,
    }
    ctx.state_events.append(event)
    logger.debug("harness transition %s -> %s (run=%s)", event["from_state"], event["to_state"], ctx.run_id)


def _persist_audit_events(ctx: HarnessContext) -> None:
    """将**新增的**状态转移审计事件追加写入 ``data/traces/<日期>/<run_id>.audit.jsonl``。

    增量落盘：用 ``ctx._audit_flushed_count`` 记录已经写过的条数，多次调用（如
    ``_handle_persist`` 内一次 + 状态机循环结束后一次）不会重复写入同一批事件。
    失败不阻断主流程（try/except + warning log），与 src.trace 的落盘策略一致。
    """
    pending = ctx.state_events[ctx._audit_flushed_count:]
    if not pending:
        return
    try:
        from datetime import date

        from src.db.engine import append_jsonl
        from src.trace import traces_root

        day_dir = traces_root() / date.today().isoformat()
        filepath = day_dir / f"{_safe_name(ctx.run_id)}.audit.jsonl"
        for event in pending:
            append_jsonl(filepath, event)
        ctx._audit_flushed_count = len(ctx.state_events)
    except Exception as e:  # noqa: BLE001 - 审计落盘失败非致命，不阻断主流程
        logger.warning("状态转移审计事件落盘失败（非致命）: %s", e)


def _export_metrics(ctx: HarnessContext) -> None:
    """导出本次 run 的轻量指标（供外部采集，不引入 Prometheus 依赖）。

    追加写入 ``data/traces/<日期>/metrics.jsonl``（一天一个文件，多个 run 共享）。
    失败不阻断主流程。
    """
    try:
        from datetime import date

        from src.db.engine import append_jsonl
        from src.trace import traces_root

        _sync_latency(ctx)
        metrics = {
            "run_id": ctx.run_id,
            "route": ctx.route_decision.route if ctx.route_decision else "unknown",
            "risk_level": ctx.risk_level,
            "total_latency_ms": ctx.budget_tracker.latency_ms,
            "tool_calls": ctx.budget_tracker.tool_calls_used,
            "rewrite_count": ctx.rewrite_count,
            "grounded": (ctx.eval_result or {}).get("passed") if ctx.eval_result else None,
            "needs_human": ctx.needs_human,
            "prompt_version": _prompt_version_hash(),
            "timestamp": time.time(),
        }
        day_dir = traces_root() / date.today().isoformat()
        append_jsonl(day_dir / "metrics.jsonl", metrics)
    except Exception as e:  # noqa: BLE001 - metrics 导出失败非致命
        logger.warning("metrics 导出失败（非致命）: %s", e)


async def _persist_run(ctx: HarnessContext, *, status: str, final_answer: str | None,
                       final_action: str | None, needs_human: bool, total_latency: float) -> None:
    """落盘 AgentRun + 所有 AgentStep（JSONL，D-12 附），并把结果写回 ctx.result_*。"""
    from src.db.engine import get_sessionmaker
    run = ctx.run
    run.status = status
    run.final_answer = final_answer
    run.final_action = final_action
    run.human_review_required = needs_human
    run.total_steps = len(ctx.steps)
    run.total_tool_calls = ctx.total_tool_calls
    run.total_latency_ms = total_latency
    run.tool_error_count = ctx.tool_error_count
    run.permission_deny_count = ctx.perm_deny_count
    run.errors_json = json.dumps(ctx.errors, ensure_ascii=False) if ctx.errors else None
    run.approvals_json = json.dumps(ctx.approvals, ensure_ascii=False) if ctx.approvals else None
    run.audit_trace_json = json.dumps(ctx.audit, ensure_ascii=False)
    async with get_sessionmaker()() as session:
        session.add(run)
        for step in ctx.steps:
            session.add(step)
        await session.commit()
    ctx.result_status = status
    ctx.result_final_answer = final_answer
    ctx.result_final_action = final_action
    ctx.result_needs_human = needs_human
    ctx.result_latency_ms = total_latency


def _build_result(ctx: HarnessContext) -> AgentHarnessResult:
    """从 ctx 组装对外结果（字段与重构前 AgentHarnessResult 完全一致）。"""
    return AgentHarnessResult(
        run_id=ctx.run_id,
        status=ctx.result_status,
        final_answer=ctx.result_final_answer,
        final_action=ctx.result_final_action,
        human_review_required=ctx.result_needs_human,
        total_steps=len(ctx.steps),
        total_tool_calls=ctx.total_tool_calls,
        total_latency_ms=ctx.result_latency_ms,
        tool_error_count=ctx.tool_error_count,
        permission_deny_count=ctx.perm_deny_count,
        errors=ctx.errors,
        approvals=ctx.approvals,
        audit_trace=ctx.audit,
        evidence=_extract_evidence(ctx.observations),
    )


async def _build_failed(ctx: HarnessContext, exc: Exception) -> AgentHarnessResult:
    """顶层异常兜底：落 failed 状态并持久化（等价于旧 harness 的 except 块）。"""
    from src.db.engine import get_sessionmaker
    total_latency = (time.perf_counter() - ctx.t_start) * 1000
    ctx.errors.append({"error": str(exc), "type": type(exc).__name__})
    run = ctx.run
    run.status = "failed"
    run.termination_reason = str(exc)[:128]
    run.total_steps = len(ctx.steps)
    run.total_tool_calls = ctx.total_tool_calls
    run.total_latency_ms = total_latency
    run.tool_error_count = ctx.tool_error_count + 1
    run.errors_json = json.dumps(ctx.errors, ensure_ascii=False)
    run.audit_trace_json = json.dumps(ctx.audit, ensure_ascii=False)
    async with get_sessionmaker()() as session:
        session.add(run)
        for step in ctx.steps:
            session.add(step)
        await session.commit()
    # 崩溃路径也尽量把已经记录的状态转移落盘（best effort，内部已自带 try/except）
    _persist_audit_events(ctx)
    _export_metrics(ctx)
    return AgentHarnessResult(
        run_id=ctx.run_id,
        status="failed",
        total_steps=len(ctx.steps),
        total_tool_calls=ctx.total_tool_calls,
        total_latency_ms=total_latency,
        tool_error_count=ctx.tool_error_count + 1,
        errors=ctx.errors,
        audit_trace=ctx.audit,
    )


# ── Harness V2 Phase 2: CHECKPOINT 断点 / 幂等续跑 ──────────────────────
#
# 每次工具调用**成功后**记一条 Checkpoint（step_index + 参数指纹 + 结果指纹）。
# 中断重跑时 _should_skip_step 用 (step_index, tool_args_hash, result_hash!=None)
# 判定该步是否已成功执行过，命中则跳过——实现幂等续跑（Phase 4 落盘后跨进程生效）。

def _checkpoint_args_hash(tool_name: str, params: dict[str, Any]) -> str:
    """工具参数指纹：sha256(tool_name:sorted(params.items()))[:16]。"""
    try:
        items = sorted(params.items())
    except Exception:  # noqa: BLE001 - 参数含不可比较类型时退化到字符串排序
        items = sorted((str(k), str(v)) for k, v in params.items())
    return hashlib.sha256(f"{tool_name}:{items}".encode("utf-8")).hexdigest()[:16]


def _checkpoint_result_hash(result: Any) -> str:
    """工具结果指纹：sha256(json.dumps(result))[:16]。"""
    return hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _record_checkpoint(ctx: HarnessContext, step_index: int, tool_name: str,
                       params: dict[str, Any], result: Any) -> Checkpoint:
    """记录一条成功执行的断点快照到 ctx.checkpoints。"""
    cp: Checkpoint = {
        "run_id": ctx.run_id,
        "step_index": step_index,
        "tool_name": tool_name,
        "tool_args_hash": _checkpoint_args_hash(tool_name, params),
        "result_hash": _checkpoint_result_hash(result),
        "state": HarnessState.EXECUTE_LOOP.value,
        "timestamp": time.time(),
    }
    ctx.checkpoints.append(cp)
    return cp


def _should_skip_step(ctx: HarnessContext, step_index: int, tool_name: str,
                      params: dict[str, Any]) -> bool:
    """检查 checkpoint：若该步骤（同 index + 同参数指纹）已成功执行过则跳过。

    幂等续跑的核心判据：只有 result_hash 非空（= 上次确实拿到了成功结果）才算数，
    避免把“记录过但失败”的步骤误判为已完成。
    """
    args_hash = _checkpoint_args_hash(tool_name, params)
    return any(
        cp.get("step_index") == step_index
        and cp.get("tool_args_hash") == args_hash
        and cp.get("result_hash") is not None
        for cp in ctx.checkpoints
    )


async def _run_plan_step(ctx: HarnessContext, i: int, step_def: dict) -> str:
    """执行计划中的**一步**（搬运自旧 harness 执行循环体，行为保持一致）。

    返回信号（Phase 2 扩展）：
      - ``"break"``          ：触发预算/循环上限，应离开执行循环 → DRAFT
      - ``"continue"``       ：本步处理完毕，进入 CHECKPOINT 后继续下一步
      - ``"hitl_action"``    ：工具需人工动作审批 → HITL_ACTION（记录并跳过该步）
      - ``"error_classify"`` ：权限硬拒绝等 → ERROR_CLASSIFY（SKIP 跳过该步）

    工具级失败三分类 + 指数退避重试（BLUEPRINT D-13）仍内联在此，沿用已验证实现，
    不拆分到 ERROR_CLASSIFY/RETRY_BACKOFF（那两个状态负责计划级/状态机级错误）。
    """
    import asyncio
    registry = ctx.registry

    if ctx.step_index >= ctx.max_steps:
        ctx.errors.append({"step": i, "error": "max_steps_exceeded"})
        return "break"

    elapsed_ms = (time.perf_counter() - ctx.t_start) * 1000
    if elapsed_ms > ctx.max_latency_ms:
        ctx.errors.append({"step": i, "error": "max_latency_exceeded", "elapsed_ms": elapsed_ms})
        return "break"

    ctx.transition_count += 1
    if ctx.transition_count > ctx.max_transitions:
        ctx.errors.append({"step": i, "error": "max_transitions_exceeded"})
        return "break"

    step_type = step_def.get("type", "execute")
    tool_name = step_def.get("tool")

    if step_type == "retrieve":
        if ctx.total_tool_calls >= ctx.max_tool_calls:
            ctx.errors.append({"step": i, "error": "max_tool_calls_exceeded"})
            return "break"
        rq = step_def.get("query") or step_def.get("description") or ctx.objective
        ls_params = {"query": rq, "top_k": 5}
        # CHECKPOINT 幂等续跑：该 retrieve 步已成功执行过则跳过（不重复检索）
        if _should_skip_step(ctx, i, "local_search", ls_params):
            ctx.audit.append({"step": "checkpoint_skip", "index": i, "tool": "local_search"})
            ctx.set_audit_meta(tool_name="local_search", tool_result_status="skipped_checkpoint")
            return "continue"
        # 工具调用前：记录 tool_name / tool_args_hash（Phase 4 审计标准化字段）
        ctx.set_audit_meta(tool_name="local_search",
                           tool_args_hash=_checkpoint_args_hash("local_search", ls_params))
        t0 = time.perf_counter()
        res = None
        try:
            res = await asyncio.wait_for(
                registry.execute("local_search", ls_params,
                                 user_context=ctx.user_context, tenant_id=ctx.tenant_id),
                timeout=ctx.step_timeout,
            )
        except Exception as e:  # noqa: BLE001
            ctx.errors.append({"step": i, "tool": "local_search", "error": f"{type(e).__name__}: {e}"})
        ctx.total_tool_calls += 1
        ctx.budget_tracker.tool_calls_used += 1
        lat = (time.perf_counter() - t0) * 1000
        if res is not None and res.success:
            snippets = (res.data or {}).get("snippets", [])
            sig = _sig([s.get("citation_id") for s in snippets])
            ctx.seen_obs_sigs[sig] = ctx.seen_obs_sigs.get(sig, 0) + 1
            ctx.observations.append({"tool": "local_search", "result": res.data})
            ctx.add_step("retrieve", {"query": rq}, {"count": len(snippets)},
                         tool_name="local_search", latency=lat)
            ctx.audit.append({"step": "retrieve", "query": rq[:60], "count": len(snippets),
                              "sig_repeat": ctx.seen_obs_sigs[sig]})
            ctx.set_audit_meta(tool_name="local_search", tool_result_status="success",
                               tool_latency_ms=int(lat))
            ctx.budget_tracker.tokens_estimated += _estimate_tokens(res.data)
            # 成功后记录断点（供中断续跑跳过）
            _record_checkpoint(ctx, i, "local_search", ls_params, res.data)
            if ctx.seen_obs_sigs[sig] > ctx.loop_detect_threshold:
                ctx.loop_detected = True
                ctx.run.termination_reason = "loop_detected:repeated_retrieval"
                ctx.audit.append({"step": "loop_detected", "loop_detected": True,
                                  "where": "retrieve", "sig": sig})
                return "break"
        else:
            err = (res.error if res is not None else "no_result")
            ctx.add_step("retrieve", {"query": rq}, {"error": err},
                         tool_name="local_search", latency=lat, error=err)
            ctx.errors.append({"step": i, "tool": "local_search", "error": err})
            # 工具调用后（失败路径同样要落 status/latency，否则审计里这两字段会缺失）
            ctx.set_audit_meta(tool_name="local_search", tool_result_status="error",
                               tool_latency_ms=int(lat))
        return "continue"

    if step_type == "execute" and tool_name:
        if ctx.total_tool_calls >= ctx.max_tool_calls:
            ctx.errors.append({"step": i, "error": "max_tool_calls_exceeded"})
            return "break"

        tool = registry.get(tool_name)
        if tool is None:
            ctx.add_step("execute", {"tool": tool_name}, {}, tool_name=tool_name,
                         error=f"Unknown tool: {tool_name}")
            ctx.tool_error_count += 1
            ctx.audit.append({"step": "execute", "tool": tool_name, "error": "unknown_tool"})
            ctx.set_audit_meta(tool_name=tool_name, tool_result_status="unknown_tool")
            return "continue"

        from src.agent.permission_gate import check_permission
        params = _prepare_step_params(tool_name, step_def, ctx.observations, ctx.objective)
        # CHECKPOINT 幂等续跑：该 execute 步已成功执行过则跳过（不重复产生副作用）
        if _should_skip_step(ctx, i, tool_name, params):
            ctx.audit.append({"step": "checkpoint_skip", "index": i, "tool": tool_name})
            ctx.set_audit_meta(tool_name=tool_name, tool_result_status="skipped_checkpoint")
            return "continue"
        perm = check_permission(tool, ctx.user_context, params, ctx.tenant_id)

        if perm.needs_approval:
            # Phase 2, 2.2：需人工动作审批 → 暂存待审批动作，转 HITL_ACTION（记录并跳过该步）。
            # 真实异步暂停/恢复需 WebSocket/回调，不在本期范围。
            ctx.pending_hitl = {"tool": tool_name, "params": params,
                                "reason": perm.reason, "risk_level": perm.risk_level}
            ctx.add_step("execute", {"tool": tool_name, "params": params}, {},
                         tool_name=tool_name, tool_params=params,
                         permission="needs_approval", error=perm.reason)
            ctx.audit.append({"step": "execute", "tool": tool_name,
                              "permission": "needs_approval", "reason": perm.reason})
            ctx.set_audit_meta(tool_name=tool_name, tool_result_status="needs_approval")
            return "hitl_action"

        if not perm.allowed:
            # 硬拒绝（无审批通道）→ ERROR_CLASSIFY(SKIP)：记审计后跳过该步继续
            ctx.perm_deny_count += 1
            ctx.add_step("execute", {"tool": tool_name, "params": params}, {},
                         tool_name=tool_name, tool_params=params,
                         permission="denied", error=perm.reason)
            ctx.audit.append({"step": "execute", "tool": tool_name, "permission": "denied",
                              "reason": perm.reason})
            ctx.set_audit_meta(tool_name=tool_name, tool_result_status="permission_denied")
            ctx.error_origin_state = HarnessState.EXECUTE_LOOP
            ctx.error_detail = {"kind": "permission_denied", "reason": perm.reason}
            ctx.error_action = ErrorAction.SKIP
            return "error_classify"

        max_attempts = 1 + max(0, tool.max_retries)
        attempt = 0
        # 工具调用前：记录 tool_name / tool_args_hash（Phase 4 审计标准化字段）
        ctx.set_audit_meta(tool_name=tool_name,
                           tool_args_hash=_checkpoint_args_hash(tool_name, params))
        while True:
            attempt += 1
            t0 = time.perf_counter()
            result = None
            exc_msg: str | None = None
            try:
                result = await asyncio.wait_for(
                    registry.execute(tool_name, params,
                                     user_context=ctx.user_context, tenant_id=ctx.tenant_id),
                    timeout=ctx.step_timeout,
                )
            except asyncio.TimeoutError:
                exc_msg = f"harness_step_timeout ({ctx.step_timeout}s)"
            except Exception as e:  # noqa: BLE001
                exc_msg = f"{type(e).__name__}: {e}"
            latency = (time.perf_counter() - t0) * 1000
            ctx.total_tool_calls += 1
            ctx.budget_tracker.tool_calls_used += 1

            succeeded = result is not None and result.success
            err_text = exc_msg or (result.error if result is not None else "unknown_error")
            classification = "success" if succeeded else _classify_tool_failure(result, exc_msg)

            ctx.add_step(
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
                ctx.observations.append({"tool": tool_name, "result": result.data})
                ctx.audit.append({"step": "execute", "tool": tool_name, "success": True, "attempt": attempt})
                ctx.set_audit_meta(tool_name=tool_name, tool_result_status="success",
                                   tool_latency_ms=int(latency))
                ctx.budget_tracker.tokens_estimated += _estimate_tokens(result.data)
                # 成功后记录断点（供中断续跑跳过，避免重复副作用）
                _record_checkpoint(ctx, i, tool_name, params, result.data)
                break

            ctx.audit.append({"step": "execute", "tool": tool_name, "error": err_text,
                              "classification": classification, "attempt": attempt})

            if (classification == "retryable" and attempt < max_attempts
                    and ctx.total_tool_calls < ctx.max_tool_calls):
                backoff = min(2 ** (attempt - 1), 8) * 0.1
                ctx.audit.append({"step": "retry", "tool": tool_name,
                                  "attempt": attempt + 1, "backoff_s": round(backoff, 2)})
                await asyncio.sleep(backoff)
                continue

            ctx.tool_error_count += 1
            ctx.errors.append({"step": i, "tool": tool_name, "error": err_text,
                               "classification": classification})
            ctx.set_audit_meta(tool_name=tool_name, tool_result_status=classification,
                               tool_latency_ms=int(latency))
            if classification == "needs_human":
                ctx.approvals.append({"tool": tool_name, "params": params,
                                      "reason": "tool_failure_needs_human"})
            break

        ctx.add_step("observe", {"observations": ctx.observations[-3:] if ctx.observations else []},
                     {"count": len(ctx.observations)})
        return "continue"

    # 既非 retrieve、也非带 tool 的 execute（兜底观测记录，与旧逻辑一致）
    ctx.add_step("observe", {"observations": ctx.observations[-3:] if ctx.observations else []},
                 {"count": len(ctx.observations)})
    return "continue"


# ── Harness V2: 状态 handler（每个只做一件事，返回下一个状态）──────────────

async def _handle_init(ctx: HarnessContext) -> HarnessState:
    """INIT：准备工具注册表，进入安全检查。"""
    from src.agent.tool_registry import get_tool_registry
    ctx.registry = get_tool_registry()
    return HarnessState.SECURITY_CHECK


async def _handle_security_check(ctx: HarnessContext) -> HarnessState:
    """SECURITY_CHECK：注入/越权检测。命中 → 落盘拦截结果并 REJECTED；否则 → CONTEXT_LOAD。"""
    check = _input_injection_detection(ctx.objective)
    if check["blocked"]:
        reason = check["reason"]
        ctx.add_step("plan", {"action": "input_sanitize"}, {"blocked": True, "reason": reason})
        ctx.audit.append({"step": "input_sanitize", "blocked": True, "reason": reason})
        draft = "您的请求因安全原因被拦截：" + reason
        ctx.draft = draft
        ctx.add_step("evaluate", {"action": "draft"}, {"draft": draft})
        _record_session_turn(ctx.session_id, ctx.objective, draft)
        total_latency = (time.perf_counter() - ctx.t_start) * 1000
        await _persist_run(ctx, status="completed", final_answer=draft,
                           final_action="input_blocked", needs_human=False, total_latency=total_latency)
        ctx.set_audit_meta(guard_result="blocked", guard_reason=reason)
        return HarnessState.REJECTED
    return HarnessState.CONTEXT_LOAD


def _estimate_token_remaining(ctx: HarnessContext) -> int:
    """估算当前 token 预算余量（供 Layer 2 惰性摘要的触发条件3 判断）。

    粗估：总预算 = 窗口 × EVIDENCE_BUDGET_RATIO；已消耗 ≈ (objective + 前几条观测)
    的字符数 × TOKEN_ESTIMATE_RATIO。CONTEXT_LOAD 阶段 observations 通常为空，
    这里只取前 3 条做保守估计即可。
    """
    total_budget = int(MODEL_CONTEXT_WINDOW * EVIDENCE_BUDGET_RATIO)
    used_chars = len(ctx.objective) + sum(len(str(o)) for o in ctx.observations[:3])
    used_tokens = int(used_chars * TOKEN_ESTIMATE_RATIO)
    return max(0, total_budget - used_tokens)


async def _handle_context_load(ctx: HarnessContext) -> HarnessState:
    """CONTEXT_LOAD：分层加载会话上下文（Phase 3，防御式，任何失败都不阻断主流程）。

    Layer 1 (recent_raw) 始终加载（零 LLM 成本）；Layer 2 (lazy_summary) 仅在
    历史超阈值 + 当前输入含指代 + token 预算有余量时惰性生成。
    """
    ctx.context_layers_used = []

    if not ctx.session_id:
        ctx.session_raw = ""
        ctx.session_summary = ""
        ctx.set_audit_meta(session_id="", context_layers_used=[])
        return HarnessState.RISK_INTENT

    try:
        from src.session_mgr import get_session
        mem = get_session(ctx.session_id)
        if mem is None or mem.is_empty():
            ctx.session_raw = ""
            ctx.session_summary = ""
            ctx.set_audit_meta(session_id=ctx.session_id or "", context_layers_used=[])
            return HarnessState.RISK_INTENT

        # Layer 1: 最近原文（始终加载）
        ctx.session_raw = mem.get_recent_raw()
        if ctx.session_raw:
            ctx.context_layers_used.append("raw_recent")

        # Layer 2: 惰性摘要（按条件触发）
        token_remaining = _estimate_token_remaining(ctx)
        ctx.session_summary = mem.get_or_generate_summary(ctx.objective, token_remaining)
        if ctx.session_summary:
            ctx.context_layers_used.append("lazy_summary")

        ctx.set_audit_meta(session_id=ctx.session_id or "",
                           context_layers_used=list(ctx.context_layers_used))
    except Exception as e:  # noqa: BLE001 - 上下文加载失败非致命
        logger.warning("Context load failed (non-critical): %s", e)
        ctx.session_raw = ""
        ctx.session_summary = ""
        ctx.set_audit_meta(session_id=ctx.session_id or "", context_layers_used=[])

    return HarnessState.RISK_INTENT


async def _handle_risk_intent(ctx: HarnessContext) -> HarnessState:
    """RISK_INTENT：风险评估 + 意图分类（合并旧 Step1/Step1.5），产出 RouteDecision。"""
    ctx.add_step("plan", {"action": "risk_pre_check"}, {"risk_level": ctx.risk_level})
    ctx.audit.append({"step": "risk_pre_check", "risk_level": ctx.risk_level})
    if ctx.session_summary:
        ctx.audit.append({"step": "session_summary", "injected": True,
                          "chars": len(ctx.session_summary)})
    ctx.intent_info = _classify_intent(ctx.objective)
    ctx.add_step("plan", {"action": "intent_classify"}, ctx.intent_info)
    ctx.audit.append({"step": "intent_classify", **ctx.intent_info})
    ctx.route_decision = _decide_route(ctx)
    # Phase 4：把本次路由档位对应的 RouteBudget 同步给 budget_tracker（此前是 BUDGET_FULL 默认值）
    ctx.budget_tracker.budget = dict(ctx.route_decision.budget)
    ctx.set_audit_meta(
        risk_level=ctx.risk_level,
        intent=ctx.intent_info.get("intent", ""),
        route_decision={
            "route": ctx.route_decision.route,
            "risk_level": ctx.route_decision.risk_level,
            "intent": ctx.route_decision.intent,
        },
    )
    return HarnessState.ROUTE


async def _handle_route(ctx: HarnessContext) -> HarnessState:
    """ROUTE：按 RouteDecision.route 分发到 fast / full / hitl_pre / reject。"""
    # Phase 2, 2.6：ROUTE 阶段前校验 UserContext 结构；非法 → 鉴权失败拒绝
    from src.agent.permission_gate import validate_user_context
    if ctx.user_context and not validate_user_context(ctx.user_context):
        ctx.reject_reason = "鉴权失败：user_context 无效"
        draft = "鉴权失败：user_context 无效，请求已被拒绝执行。"
        ctx.draft = draft
        ctx.add_step("approve", {"action": "validate_user_context"}, {"valid": False})
        ctx.audit.append({"step": "route", "rejected": True, "reason": ctx.reject_reason})
        total_latency = (time.perf_counter() - ctx.t_start) * 1000
        await _persist_run(ctx, status="completed", final_answer=draft, final_action="rejected",
                           needs_human=False, total_latency=total_latency)
        ctx.set_audit_meta(error_class="invalid_user_context",
                           error_action=ErrorAction.ABORT.value,
                           route_decision={"route": "reject"})
        return HarnessState.REJECTED
    rd = ctx.route_decision
    route = rd.route if rd is not None else "full"
    if route == "fast":
        ctx.from_fast = True
        _apply_route_budget(ctx, BUDGET_FAST)
        return HarnessState.FAST_PATH
    if route == "hitl_pre":
        _apply_route_budget(ctx, BUDGET_HITL)
        return HarnessState.HITL_PRE
    if route == "reject":
        draft = "您的请求因合规原因被拒绝执行。"
        ctx.draft = draft
        total_latency = (time.perf_counter() - ctx.t_start) * 1000
        await _persist_run(ctx, status="completed", final_answer=draft, final_action="rejected",
                           needs_human=False, total_latency=total_latency)
        return HarnessState.REJECTED
    # full：保留调用方预算（_apply_caller_budget 已设置），不覆盖
    return HarnessState.PLAN


async def _handle_fast_path(ctx: HarnessContext) -> HarnessState:
    """FAST_PATH：跳过规划器，直接一次检索 + 生成草稿，再过护栏。"""
    import asyncio
    registry = ctx.registry
    rq = ctx.objective
    ls_params = {"query": rq, "top_k": 5}
    # 工具调用前：记录 tool_name / tool_args_hash（Phase 4 审计标准化字段）
    ctx.set_audit_meta(tool_name="local_search",
                       tool_args_hash=_checkpoint_args_hash("local_search", ls_params))
    t0 = time.perf_counter()
    res = None
    try:
        res = await asyncio.wait_for(
            registry.execute("local_search", ls_params,
                             user_context=ctx.user_context, tenant_id=ctx.tenant_id),
            timeout=ctx.step_timeout)
    except Exception as e:  # noqa: BLE001
        ctx.errors.append({"step": "fast_path", "tool": "local_search",
                           "error": f"{type(e).__name__}: {e}"})
    ctx.total_tool_calls += 1
    ctx.budget_tracker.tool_calls_used += 1
    lat = (time.perf_counter() - t0) * 1000
    if res is not None and res.success:
        snippets = (res.data or {}).get("snippets", [])
        ctx.observations.append({"tool": "local_search", "result": res.data})
        ctx.add_step("retrieve", {"query": rq, "path": "fast"}, {"count": len(snippets)},
                     tool_name="local_search", latency=lat)
        ctx.audit.append({"step": "fast_path_retrieve", "query": rq[:60], "count": len(snippets)})
        ctx.set_audit_meta(tool_name="local_search", tool_result_status="success",
                           tool_latency_ms=int(lat))
        ctx.budget_tracker.tokens_estimated += _estimate_tokens(res.data)
    else:
        err = (res.error if res is not None else "no_result")
        ctx.add_step("retrieve", {"query": rq, "path": "fast"}, {"error": err},
                     tool_name="local_search", latency=lat, error=err)
        ctx.errors.append({"step": "fast_path", "tool": "local_search", "error": err})
        ctx.set_audit_meta(tool_name="local_search", tool_result_status="error",
                           tool_latency_ms=int(lat))
    ctx.draft = _generate_draft(ctx.objective, ctx.observations, ctx.errors,
                                session_summary=ctx.session_summary, session_raw=ctx.session_raw)
    ctx.add_step("evaluate", {"action": "draft", "path": "fast"}, {"draft": ctx.draft})
    ctx.audit.append({"step": "fast_path_draft", "length": len(ctx.draft)})
    return HarnessState.OUTPUT_GUARD


async def _handle_plan(ctx: HarnessContext) -> HarnessState:
    """PLAN：复用 _build_plan（三层 fallback），产出执行计划。"""
    plan = _build_plan(ctx.objective, ctx.risk_level, intent_info=ctx.intent_info)
    ctx.plan = plan
    ctx.add_step("plan", {"action": "build_plan", "objective": ctx.objective}, {"plan": plan})
    ctx.run.plan_json = json.dumps(plan, ensure_ascii=False)
    ctx.audit.append({"step": "plan", "steps": len(plan)})
    return HarnessState.PLAN_VALIDATE


async def _handle_plan_validate(ctx: HarnessContext) -> HarnessState:
    """PLAN_VALIDATE：pydantic 校验计划；非法 → ERROR_CLASSIFY（降级到规则兜底）。"""
    validated, errors = _validate_plan(ctx.plan)
    if validated is None:
        ctx.error_origin_state = HarnessState.PLAN_VALIDATE
        ctx.error_detail = {"kind": "plan_invalid", "errors": errors}
        return HarnessState.ERROR_CLASSIFY
    ctx.plan = validated
    ctx.plan_cursor = 0
    return HarnessState.EXECUTE_LOOP


async def _handle_execute_loop(ctx: HarnessContext) -> HarnessState:
    """EXECUTE_LOOP：执行计划游标处的一步。

    信号映射（Phase 2）：break→DRAFT；hitl_action→HITL_ACTION；
    error_classify→ERROR_CLASSIFY；continue→CHECKPOINT；游标耗尽→DRAFT。

    Phase 4：进入本状态先做一次 BudgetTracker 软护栏检查——任一维度超预算
    （steps/tool_calls/tokens/latency/transitions）即转 DEGRADED，用已有证据降级收尾，
    与 _run_plan_step 内旧的 per-step 预算检查（返回 "break"→DRAFT）是并行的两层护栏。
    """
    if ctx.plan_cursor >= len(ctx.plan):
        return HarnessState.DRAFT
    _sync_latency(ctx)
    if ctx.budget_tracker.is_over_budget() and not ctx.degraded_forced:
        ctx.degraded_forced = True
        snapshot = ctx.budget_tracker.snapshot()
        ctx.errors.append({"error": "budget_exceeded", "budget_consumed": snapshot})
        ctx.audit.append({"step": "budget_exceeded", "budget_consumed": snapshot})
        ctx.set_audit_meta(error_class="budget_exceeded", error_action=ErrorAction.DEGRADE.value,
                           budget_consumed=snapshot)
        return HarnessState.DEGRADED
    i = ctx.plan_cursor
    signal = await _run_plan_step(ctx, i, ctx.plan[i])
    ctx.budget_tracker.steps_used += 1
    if signal == "break":
        return HarnessState.DRAFT
    if signal == "hitl_action":
        return HarnessState.HITL_ACTION
    if signal == "error_classify":
        return HarnessState.ERROR_CLASSIFY
    return HarnessState.CHECKPOINT


async def _handle_checkpoint(ctx: HarnessContext) -> HarnessState:
    """CHECKPOINT：步骤游标 +1 后回到执行循环。

    Phase 2：断点快照的记录已下沉到 _run_plan_step（仅在工具调用**成功后**记录，
    带 sha256 参数/结果指纹，供 _should_skip_step 幂等续跑）；本状态只负责推进游标，
    避免对失败/跳过/需审批的步骤也记录断点。

    Phase 4：把刚完成步骤的 step_index / result_hash 写入本次转移的审计元数据
    （result_hash 仅在该步确实记录了断点时非空）。
    """
    step_index = ctx.plan_cursor
    cp = next((c for c in reversed(ctx.checkpoints) if c.get("step_index") == step_index), None)
    ctx.set_audit_meta(step_index=step_index, result_hash=(cp or {}).get("result_hash"))
    ctx.plan_cursor += 1
    return HarnessState.EXECUTE_LOOP


async def _handle_draft(ctx: HarnessContext) -> HarnessState:
    """DRAFT：复用 _generate_draft 基于证据生成草稿 → OUTPUT_GUARD。"""
    ctx.draft = _generate_draft(ctx.objective, ctx.observations, ctx.errors,
                                session_summary=ctx.session_summary, session_raw=ctx.session_raw)
    ctx.add_step("evaluate", {"action": "draft"}, {"draft": ctx.draft})
    ctx.audit.append({"step": "draft", "length": len(ctx.draft)})
    return HarnessState.OUTPUT_GUARD


async def _handle_output_guard(ctx: HarnessContext) -> HarnessState:
    """OUTPUT_GUARD：输出护栏 + 闭环路由。

    入口来源（DRAFT 首次 / REWRITE 回环 / FAST_PATH / DEGRADED）通过上一条状态转移
    审计事件识别，仅用于可观测；路由规则（Phase 2）：
      - 护栏拦截（blocked）→ HITL_OUTPUT（高风险输出需人工确认，2.3）
      - FAST_PATH 来源 → PERSIST（跳过评估）
      - 其余（DRAFT / REWRITE）→ EVALUATE（REWRITE→OUTPUT_GUARD→EVALUATE 闭环，2.1）
    """
    entry = ctx.state_events[-1]["from_state"] if ctx.state_events else HarnessState.DRAFT.value
    ctx.output_guard_entry = entry
    blocked = False
    threats: list[str] = []
    try:
        from src.input_sanitizer import OutputGuard
        check = OutputGuard.check(ctx.draft)
        if check.blocked:
            ctx.draft = check.sanitized
            blocked = True
            threats = list(check.threats)
            ctx.set_audit_meta(guard_result="blocked", guard_reason=str(check.threats))
        else:
            # 放行也写 guard_reason（空串），让下游聚合无需处理字段缺失
            ctx.set_audit_meta(guard_result="pass", guard_reason="")
    except Exception as e:  # noqa: BLE001
        logger.debug("OutputGuard check failed (non-critical): %s", e)
        ctx.set_audit_meta(guard_result="error", guard_reason=f"{type(e).__name__}: {e}")
    ctx.guard_blocked = blocked  # 每次进护栏都重置，避免上一次拦截状态残留
    ctx.audit.append({"step": "output_guard", "entry": entry, "blocked": blocked, "threats": threats})
    if blocked:
        return HarnessState.HITL_OUTPUT
    if ctx.from_fast:
        return HarnessState.PERSIST
    return HarnessState.EVALUATE


async def _handle_evaluate(ctx: HarnessContext) -> HarnessState:
    """EVALUATE：复用 _evaluate_result。pass → HITL_OUTPUT；可改写 → REWRITE；耗尽 → HITL_OUTPUT。"""
    ctx.eval_result = _evaluate_result(ctx.draft, ctx.observations, ctx.errors)
    passed = ctx.eval_result.get("passed", True)
    if ctx.rewrite_count == 0:
        ctx.add_step("evaluate", {"action": "evaluate"}, ctx.eval_result)
        ctx.audit.append({"step": "evaluate", "passed": passed})
    else:
        ctx.add_step("evaluate", {"action": f"rewrite_evaluate_{ctx.rewrite_count}"}, ctx.eval_result)
        ctx.audit.append({"step": f"rewrite_{ctx.rewrite_count}", "passed": passed,
                          "obs_count": len(ctx.observations)})
    ctx.set_audit_meta(eval_grounded=passed, eval_issues=list(ctx.eval_result.get("issues", [])),
                       rewrite_count=ctx.rewrite_count)
    if passed:
        return HarnessState.HITL_OUTPUT
    if ctx.rewrite_count < ctx.max_rewrite_attempts and not ctx.loop_detected:
        return HarnessState.REWRITE
    if ctx.rewrite_count >= ctx.max_rewrite_attempts:
        ctx.audit.append({"step": "rewrite_exhausted", "forced_pass": True})
    return HarnessState.HITL_OUTPUT


async def _handle_rewrite(ctx: HarnessContext) -> HarnessState:
    """REWRITE：带评估反馈重新检索（真回环）。无新证据 → 判循环 → HITL_OUTPUT；否则重生成 → OUTPUT_GUARD。"""
    import asyncio
    ctx.rewrite_count += 1
    ctx.budget_tracker.rewrite_retries_used = ctx.rewrite_count
    registry = ctx.registry
    feedback = "；".join(str(x) for x in (ctx.eval_result or {}).get("issues", []))
    augmented_query = f"{ctx.objective} {feedback}".strip()
    obs_before = len(ctx.observations)
    new_sig = None
    if ctx.total_tool_calls < ctx.max_tool_calls:
        ls_params = {"query": augmented_query, "top_k": 5}
        # 工具调用前：记录 tool_name / tool_args_hash（Phase 4 审计标准化字段）
        ctx.set_audit_meta(tool_name="local_search",
                           tool_args_hash=_checkpoint_args_hash("local_search", ls_params))
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(
                registry.execute("local_search", ls_params,
                                 user_context=ctx.user_context, tenant_id=ctx.tenant_id),
                timeout=ctx.step_timeout)
            ctx.total_tool_calls += 1
            ctx.budget_tracker.tool_calls_used += 1
            lat = (time.perf_counter() - t0) * 1000
            if res is not None and res.success:
                snippets = (res.data or {}).get("snippets", [])
                new_sig = _sig([s.get("citation_id") for s in snippets])
                ctx.observations.append({"tool": "local_search", "result": res.data,
                                         "rewrite_round": ctx.rewrite_count})
                ctx.set_audit_meta(tool_name="local_search", tool_result_status="success",
                                   tool_latency_ms=int(lat))
                ctx.budget_tracker.tokens_estimated += _estimate_tokens(res.data)
            else:
                ctx.set_audit_meta(tool_name="local_search", tool_result_status="error",
                                   tool_latency_ms=int(lat))
        except Exception as e:  # noqa: BLE001
            ctx.errors.append({"rewrite": ctx.rewrite_count, "error": f"{type(e).__name__}: {e}"})
            ctx.set_audit_meta(tool_name="local_search", tool_result_status="exception",
                               tool_latency_ms=int((time.perf_counter() - t0) * 1000))
    ctx.add_step("retrieve", {"action": f"rewrite_retrieve_{ctx.rewrite_count}",
                              "query": augmented_query, "feedback": feedback},
                 {"obs_before": obs_before, "obs_after": len(ctx.observations)})
    no_new = len(ctx.observations) == obs_before
    repeated = new_sig is not None and ctx.seen_obs_sigs.get(new_sig, 0) > 0
    if new_sig is not None:
        ctx.seen_obs_sigs[new_sig] = ctx.seen_obs_sigs.get(new_sig, 0) + 1
    if no_new or repeated:
        ctx.loop_detected = True
        ctx.run.termination_reason = "loop_detected:no_new_evidence"
        ctx.audit.append({"step": f"rewrite_{ctx.rewrite_count}", "loop_detected": True,
                          "no_new": no_new, "repeated_sig": repeated})
        return HarnessState.HITL_OUTPUT
    # 用新证据重新生成（仍以原始问题提问，augmented_query 只用于检索），再回护栏→评估闭环
    ctx.draft = _generate_draft(ctx.objective, ctx.observations, ctx.errors,
                                session_summary=ctx.session_summary, session_raw=ctx.session_raw)
    return HarnessState.OUTPUT_GUARD


async def _handle_hitl_pre(ctx: HarnessContext) -> HarnessState:
    """HITL_PRE（Phase 2）：高风险 run 的事前人工审批。

    记录整个 run 的审批请求（objective/risk_level/reason）并标记 needs_human；
    Phase 2 不做真实异步暂停，用 _simulate_hitl_pre_approval 模拟审批决策：
      - 通过 → PLAN（进入 FULL_PIPELINE；needs_human 仍为 True，最终 waiting_approval）
      - 拒绝 → 落盘拒绝结果并 REJECTED
    """
    reason = f"risk_level={ctx.risk_level}"
    ctx.hitl_approvals.append({"type": "pre", "objective": ctx.objective[:80],
                               "risk_level": ctx.risk_level, "reason": reason})
    ctx.approvals.append({"type": "hitl_pre", "reason": reason, "objective": ctx.objective[:80]})
    ctx.needs_human = True
    ctx.hitl_type = "pre"
    ctx.add_step("approve", {"action": "hitl_pre"}, {"approvals": ctx.approvals[-1:]})
    ctx.audit.append({"step": "hitl_pre", "required": True, "risk": ctx.risk_level})

    if _simulate_hitl_pre_approval(ctx):
        # 审批通过（Phase 2 模拟）→ 进入 FULL_PIPELINE
        ctx.audit.append({"step": "hitl_pre_decision", "approved": True})
        ctx.set_audit_meta(hitl_type="pre", hitl_decision="approved")
        return HarnessState.PLAN
    # 审批拒绝：终态 REJECTED 前必须 persist，否则 result 为默认值
    ctx.reject_reason = "hitl_pre_rejected"
    ctx.audit.append({"step": "hitl_pre_decision", "approved": False})
    ctx.set_audit_meta(hitl_type="pre", hitl_decision="rejected")
    draft = "您的请求涉及高风险操作，人工审批未通过，已拒绝执行。"
    ctx.draft = draft
    total_latency = (time.perf_counter() - ctx.t_start) * 1000
    await _persist_run(ctx, status="completed", final_answer=draft, final_action="rejected",
                       needs_human=True, total_latency=total_latency)
    return HarnessState.REJECTED


async def _handle_hitl_action(ctx: HarnessContext) -> HarnessState:
    """HITL_ACTION（Phase 2）：动作级人工审批——记录并跳过该步，继续执行循环。

    真实的异步暂停/恢复需 WebSocket/回调机制，不在本期范围；Phase 2 语义为
    “记录审批请求 + 标记 needs_human + 跳过该步骤”，随后回到 EXECUTE_LOOP 处理
    下一步（计划耗尽则经 DRAFT→…→PERSIST 收尾，最终 waiting_approval）。
    """
    pending = ctx.pending_hitl or {}
    tool_name = pending.get("tool", "")
    params = pending.get("params", {})
    reason = pending.get("reason", "")
    risk_level = pending.get("risk_level", ctx.risk_level)
    ctx.hitl_approvals.append({
        "type": "action_approval", "tool_name": tool_name, "params": params,
        "risk_level": risk_level, "reason": reason,
    })
    ctx.approvals.append({"type": "hitl_action", "tool": tool_name,
                          "params": params, "reason": reason})
    ctx.needs_human = True
    ctx.hitl_type = "action_approval"
    ctx.add_step("approve", {"action": "hitl_action", "tool": tool_name},
                 {"approval": ctx.hitl_approvals[-1]})
    ctx.audit.append({"step": "hitl_action", "required": True, "tool": tool_name, "reason": reason})
    ctx.set_audit_meta(hitl_type="action_approval", hitl_decision="required", tool_name=tool_name)
    ctx.pending_hitl = None
    # 记录并跳过该步骤：游标 +1 后回到执行循环继续下一步
    ctx.plan_cursor += 1
    return HarnessState.EXECUTE_LOOP


async def _handle_hitl_output(ctx: HarnessContext) -> HarnessState:
    """HITL_OUTPUT（Phase 2）：输出级人工复核判定 → PERSIST。

    触发 needs_human 的条件（_compute_needs_human + 护栏拦截）：
      - risk_level high/critical（高风险输出即便评估通过仍需人工确认）
      - 评估多次不过（rewrite_count >= max_rewrite_attempts）
      - 护栏拦截（guard_blocked）
    命中时记录审批请求（draft 摘要 / risk_level / eval_issues / reason）并标记 hitl_type。
    """
    needs_human = _compute_needs_human(ctx) or ctx.guard_blocked
    ctx.needs_human = needs_human
    if needs_human:
        reason = _hitl_output_reason(ctx)
        eval_issues = list((ctx.eval_result or {}).get("issues", []))
        ctx.hitl_approvals.append({
            "type": "output_approval", "draft_summary": (ctx.draft or "")[:200],
            "risk_level": ctx.risk_level, "eval_issues": eval_issues, "reason": reason,
        })
        ctx.hitl_type = "output_approval"
        ctx.add_step("approve", {"action": "request_approval"},
                     {"approvals": ctx.approvals, "reason": reason})
        ctx.audit.append({"step": "hitl", "required": True,
                          "approvals": len(ctx.approvals), "reason": reason})
    else:
        ctx.audit.append({"step": "hitl", "required": False})
    ctx.set_audit_meta(hitl_type="output", hitl_decision="required" if needs_human else "auto")
    return HarnessState.PERSIST


async def _handle_error_classify(ctx: HarnessContext) -> HarnessState:
    """ERROR_CLASSIFY：按 ErrorAction 分发。计划非法优先降级到规则兜底计划。"""
    detail = ctx.error_detail or {}
    kind = detail.get("kind", "")
    if kind == "plan_invalid":
        fallback = _build_plan_rule_based(ctx.objective, ctx.risk_level)
        validated, _errs = _validate_plan(fallback)
        if validated is not None:
            ctx.plan = validated
            ctx.plan_cursor = 0
            ctx.audit.append({"step": "error_classify", "action": "degrade_plan", "kind": kind})
            ctx.set_audit_meta(error_class=kind, error_action=ErrorAction.DEGRADE.value)
            ctx.error_detail = {}
            return HarnessState.EXECUTE_LOOP
        ctx.error_action = ErrorAction.ABORT
    action = ctx.error_action or ErrorAction.DEGRADE
    action_val = getattr(action, "value", str(action))
    ctx.set_audit_meta(error_class=kind, error_action=action_val)
    ctx.audit.append({"step": "error_classify", "kind": kind, "action": action_val})
    if action == ErrorAction.RETRY:
        return HarnessState.RETRY_BACKOFF
    if action == ErrorAction.ABORT:
        return HarnessState.HITL_OUTPUT
    if action == ErrorAction.SKIP:
        ctx.plan_cursor += 1
        return HarnessState.EXECUTE_LOOP
    return HarnessState.DEGRADED


async def _handle_retry_backoff(ctx: HarnessContext) -> HarnessState:
    """RETRY_BACKOFF：指数退避后回到出错前的状态（ctx.error_origin_state）。"""
    import asyncio
    ctx.retry_attempt += 1
    backoff = min(2 ** (ctx.retry_attempt - 1), 8) * 0.1
    ctx.audit.append({"step": "retry_backoff", "attempt": ctx.retry_attempt,
                      "backoff_s": round(backoff, 2)})
    await asyncio.sleep(backoff)
    origin = ctx.error_origin_state or HarnessState.EXECUTE_LOOP
    ctx.error_detail = {}
    ctx.error_action = None
    return origin


async def _handle_degraded(ctx: HarnessContext) -> HarnessState:
    """DEGRADED：用已有证据生成降级回复，再过护栏后收尾。"""
    if not ctx.run.termination_reason:
        ctx.run.termination_reason = "degraded"
    ctx.audit.append({"step": "degraded", "reason": "budget_or_error"})
    if not ctx.draft:
        ctx.draft = _generate_draft(ctx.objective, ctx.observations, ctx.errors,
                                    session_summary=ctx.session_summary, session_raw=ctx.session_raw)
    return HarnessState.OUTPUT_GUARD


async def _handle_persist(ctx: HarnessContext) -> HarnessState:
    """PERSIST：写回会话记忆 + JSONL 落盘 + 组装结果 → DONE。"""
    total_latency = (time.perf_counter() - ctx.t_start) * 1000
    _record_session_turn(ctx.session_id, ctx.objective, ctx.draft)
    needs_human = ctx.needs_human if ctx.needs_human is not None else _compute_needs_human(ctx)
    status = "waiting_approval" if needs_human else "completed"
    final_action = "human_review_required" if needs_human else "completed"
    await _persist_run(ctx, status=status, final_answer=ctx.draft, final_action=final_action,
                       needs_human=needs_human, total_latency=total_latency)
    # Phase 4：写入 prompt 版本哈希 + 最终预算快照（PERSIST→DONE 转移自身的 metadata，
    # 由后续 _emit_audit 弹出到 state_events）；并先把已有的事件增量落盘，避免丢失。
    # metrics 不在此处导出：REJECTED 路径（鉴权失败 / HITL 预审驳回）根本不经过
    # PERSIST，而那恰恰是最需要监控的 run；改由 _run_state_machine 在循环退出后
    # 统一导出，保证「每次 run 恰好一条 metrics」且覆盖所有终态。
    _sync_latency(ctx)
    ctx.set_audit_meta(prompt_version=_prompt_version_hash(),
                       budget_consumed=ctx.budget_tracker.snapshot())
    _persist_audit_events(ctx)
    return HarnessState.DONE


#: 状态 → handler 分发表（终态 DONE/REJECTED 无 handler，while 循环遇终态即退出）。
_STATE_HANDLERS: dict[HarnessState, Callable] = {
    HarnessState.INIT: _handle_init,
    HarnessState.SECURITY_CHECK: _handle_security_check,
    HarnessState.CONTEXT_LOAD: _handle_context_load,
    HarnessState.RISK_INTENT: _handle_risk_intent,
    HarnessState.ROUTE: _handle_route,
    HarnessState.FAST_PATH: _handle_fast_path,
    HarnessState.PLAN: _handle_plan,
    HarnessState.PLAN_VALIDATE: _handle_plan_validate,
    HarnessState.EXECUTE_LOOP: _handle_execute_loop,
    HarnessState.DRAFT: _handle_draft,
    HarnessState.OUTPUT_GUARD: _handle_output_guard,
    HarnessState.EVALUATE: _handle_evaluate,
    HarnessState.REWRITE: _handle_rewrite,
    HarnessState.HITL_PRE: _handle_hitl_pre,
    HarnessState.HITL_ACTION: _handle_hitl_action,
    HarnessState.HITL_OUTPUT: _handle_hitl_output,
    HarnessState.ERROR_CLASSIFY: _handle_error_classify,
    HarnessState.RETRY_BACKOFF: _handle_retry_backoff,
    HarnessState.DEGRADED: _handle_degraded,
    HarnessState.CHECKPOINT: _handle_checkpoint,
    HarnessState.PERSIST: _handle_persist,
}


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
    """Execute an agent run as an explicit state machine (Harness V2, Phase 1).

    控制流：``INIT → SECURITY_CHECK → CONTEXT_LOAD → RISK_INTENT → ROUTE`` 后按
    路由分发到 FAST_PATH / PLAN(→EXECUTE_LOOP) / HITL_PRE，最终经 OUTPUT_GUARD /
    EVALUATE / HITL_OUTPUT 落到 PERSIST → DONE。每个节点是一个 handler，只做一件
    事并返回下一个状态；所有中间态承载在 ``HarnessContext`` 上。

    函数签名与返回类型（``AgentHarnessResult``）保持不变，向后兼容 API 层与评测脚本。

    Args:
        objective: The task objective / user query
        tenant_id: Tenant scope
        user_id: User identifier
        user_context: User roles, scopes, department
        session_id: Optional session for multi-turn
        ticket_id: Optional ticket reference
        budget: Step/tool budget limits
    """
    from src.db.models.agent_run import AgentRun

    t_start = time.perf_counter()
    eff_budget = dict(budget or DEFAULT_BUDGET)
    run_id = str(uuid.uuid4())
    user_context = user_context or {}
    risk_level = _assess_risk(objective)

    run = AgentRun(
        id=run_id,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        ticket_id=ticket_id,
        objective=objective,
        user_query=objective,
        status="running",
        risk_level=risk_level,
        budget_json=json.dumps(eff_budget, ensure_ascii=False),
    )

    ctx = HarnessContext(
        run_id=run_id,
        objective=objective,
        tenant_id=tenant_id,
        user_id=user_id,
        user_context=user_context,
        session_id=session_id,
        ticket_id=ticket_id,
        t_start=t_start,
        budget=eff_budget,
        run=run,
        risk_level=risk_level,
    )
    _apply_caller_budget(ctx)

    async def _run_state_machine() -> AgentHarnessResult:
        """显式状态机 while 循环（每个 handler 只做一件事并返回下一个状态）。"""
        state = HarnessState.INIT
        transitions = 0
        while state not in TERMINAL_STATES:
            transitions += 1
            # 状态机级转移护栏（独立于执行循环内 per-step 的 max_transitions）：
            # 只防状态机自身失控打转，正常 FULL_PIPELINE 跑不到这个上限。
            if transitions > ctx.state_transition_cap and not ctx.degraded_forced:
                ctx.degraded_forced = True
                ctx.errors.append(
                    {"error": "max_state_transitions_exceeded", "transitions": transitions}
                )
                ctx.audit.append({"step": "state_transition_cap", "transitions": transitions})
                ctx.set_audit_meta(error_class="max_state_transitions_exceeded",
                                   error_action=ErrorAction.DEGRADE.value)
                prev = state
                state = HarnessState.DEGRADED
                _emit_audit(ctx, prev, state)
                continue
            prev_state = state
            handler = _STATE_HANDLERS[state]
            state = await handler(ctx)
            _emit_audit(ctx, prev_state, state)
        # Phase 4：循环退出（DONE/REJECTED）后再做一次增量落盘，补写最后一条转移
        # 事件本身（如 PERSIST→DONE，携带 prompt_version / budget_consumed 最终快照）。
        _persist_audit_events(ctx)
        _export_metrics(ctx)
        return _build_result(ctx)

    try:
        # Phase 3：同一 session_id 的并发请求串行化，避免会话记忆读改写竞态。
        if session_id:
            from src.session_mgr import get_session_lock
            lock = get_session_lock(session_id)
            async with lock:
                return await _run_state_machine()
        return await _run_state_machine()
    except Exception as e:  # noqa: BLE001 - 顶层兜底：任何未捕获异常都落 failed 并持久化
        logger.exception("Agent harness failed: run_id=%s", run_id)
        return await _build_failed(ctx, e)


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

    三层 fallback，**任一层拿到合法计划就返回**：

    1. **native function calling**（首选）：把 ``PLAN_TOOL_SCHEMA`` 作为 ``tools=``
       传给后端，并强制 ``tool_choice`` 指向 ``build_plan``。后端返回的
       ``tool_calls[0].function.arguments`` 是 JSON 字符串，解析后得到
       ``{"steps": [...]}``。这条路径不依赖正则，格式由 schema 约束。
    2. **文本提取**（兼容）：旧 LLM 或后端不支持 tools 时，仍按纯文本
       JSON 数组返回；用 ``_extract_json_array`` 解析。
    3. **规则兜底**：LLM 不可用/两次解析都失败时，用 ``_build_plan_rule_based``。

    每条路径出来的原始计划都会过一遍 ``_validate_plan``（pydantic 强校验），
    校验失败记 warning 并继续下一层 fallback。

    函数签名与返回值（``list[dict]``）保持不变，调用方无需改动。
    """
    # 组装可用工具描述（同时用于两条路径的 system/user prompt）
    try:
        from src.agent.tool_registry import get_tool_registry
        registry = get_tool_registry()
        tools_desc = [
            f"- {t.name}: {t.description} (risk: {t.risk_level.value}, side_effect: {t.side_effect.value})"
            for t in registry.list_tools()
        ]
    except Exception as e:  # pragma: no cover - registry 不可用时不阻断规划
        logger.warning("Failed to list tools for planner prompt: %s", e)
        tools_desc = []
    tools_text = "\n".join(tools_desc) or "（无可用工具）"

    system = PLANNER_SYSTEM_PROMPT
    user = (
        f"风险等级: {risk_level}\n"
        f"可用工具:\n{tools_text}\n\n"
        f"用户目标: {objective}\n\n"
        "请生成执行计划（调用 build_plan 工具，steps 字段为步骤数组）:"
    )

    # ── 第 1 层：native function calling ──
    plan = _try_native_plan(system, user, objective)
    if plan is not None:
        return plan

    # ── 第 2 层：文本提取（向后兼容）──
    plan = _try_text_plan(system, user, objective)
    if plan is not None:
        return plan

    # ── 第 3 层：规则兜底 ──
    logger.warning("All LLM planning paths failed for '%s', using rule-based fallback", objective[:60])
    return _build_plan_rule_based(objective, risk_level)


def _try_native_plan(system: str, user: str, objective: str) -> list[dict] | None:
    """尝试用 OpenAI 兼容的 ``tools=`` / ``tool_choice=`` 拿结构化计划。

    后端不支持、未返回 tool_calls、arguments 解析失败、pydantic 校验失败时
    都返回 ``None``，由上层继续下一层 fallback。
    """
    try:
        from src.llm import chat_completion_full
        resp = chat_completion_full(
            system, user,
            tools=[PLAN_TOOL_SCHEMA],
            tool_choice=PLAN_TOOL_CHOICE,
        )
    except Exception as e:
        logger.warning("Native function-calling plan failed: %s", e)
        return None

    tool_calls = resp.get("tool_calls") or []
    if not tool_calls:
        logger.info("LLM did not return tool_calls; falling back to text extraction")
        return None

    # 找到 build_plan 的那次调用（后端可能返回多个，取第一个匹配的）
    args_raw: str | None = None
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        if fn.get("name") == PLAN_TOOL_NAME:
            args_raw = fn.get("arguments")
            break
    if args_raw is None:
        logger.info("tool_calls did not include %s; falling back", PLAN_TOOL_NAME)
        return None

    # arguments 可能是 JSON 字符串，也可能被后端直接反序列化成 dict
    parsed: Any
    if isinstance(args_raw, str):
        try:
            parsed = json.loads(args_raw)
        except json.JSONDecodeError as e:
            logger.warning("tool_calls arguments JSON decode failed: %s", e)
            return None
    else:
        parsed = args_raw

    steps = parsed.get("steps") if isinstance(parsed, dict) else None
    if not isinstance(steps, list):
        logger.warning("tool_calls arguments missing 'steps' list: %r", type(parsed).__name__)
        return None

    validated, errors = _validate_plan(steps)
    if validated is None:
        logger.warning("Native plan failed pydantic validation: %s", errors[:3])
        return None

    logger.info("LLM native plan generated: %d steps for '%s'", len(validated), objective[:50])
    return validated


def _try_text_plan(system: str, user: str, objective: str) -> list[dict] | None:
    """兼容路径：按纯文本 JSON 数组解析 LLM 输出。"""
    try:
        from src.llm import chat_completion
        raw = chat_completion(system, user)
    except Exception as e:
        logger.warning("Text-mode LLM planning failed: %s", e)
        return None

    parsed = _extract_json_array(raw)
    if not parsed:
        logger.warning("Text-mode plan parse failed (no JSON array found)")
        return None

    validated, errors = _validate_plan(parsed)
    if validated is None:
        logger.warning("Text-mode plan failed pydantic validation: %s", errors[:3])
        return None

    logger.info("LLM text-mode plan generated: %d steps for '%s'", len(validated), objective[:50])
    return validated


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
                          total_chars: int | None = None) -> str:
    """把观测里的检索片段拼成**给 LLM 看的编号证据块**。

    为什么不能 `json.dumps(result)[:500]`：local_search 的 result 是
    `{"query","retrieval_query","count","snippets":[...]}`，500 字符连第一条
    法条原文都读不到——LLM 只能凭标题猜，产出无法溯源的答案，随后 grounding
    判分必挂、回环空转（实测 D 类 15 题 grounding_rate=0.0）。
    这里直接摊平 snippets，保留 文件/标题/条号/原文，并给每条编号，便于
    生成侧写 [1][2] 引用、评估侧逐条核对。

    排序策略：所有片段按 score 降序排列后再截断，确保高分证据（无论来自
    首轮检索还是 rewrite 回环重检索）优先保留，避免新证据被预算截掉导致
    回环空转。

    ``total_chars=None`` 时由窗口配置推导（``DEFAULT_EVIDENCE_TOTAL_CHARS``，
    见模块顶部「模型上下文窗口配置」）；显式传值可覆盖（向后兼容）。
    """
    if not total_chars:
        total_chars = DEFAULT_EVIDENCE_TOTAL_CHARS
    # ① 摊平所有片段 & 综合稿，按 score 降序排序（高分优先保留）
    all_snippets: list[dict] = []
    answers: list[tuple[str, str]] = []  # (tool_name, answer_text)
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        tool = obs.get("tool", "unknown")
        result = obs.get("result")
        if not isinstance(result, dict):
            continue
        for s in (result.get("snippets") or []):
            if isinstance(s, dict) and str(s.get("text") or "").strip():
                all_snippets.append(s)
        # synthesize 等工具直接给答案文本
        ans = result.get("answer")
        if isinstance(ans, str) and ans.strip():
            answers.append((tool, ans.strip()))

    # 按 score 降序；无 score 或 None 排末尾
    all_snippets.sort(
        key=lambda s: s.get("score") if s.get("score") is not None else float('-inf'),
        reverse=True,
    )

    # ② 按预算格式化编号证据块
    lines: list[str] = []
    used = 0
    n = 0
    for s in all_snippets:
        n += 1
        text = str(s.get("text") or "").strip()
        block = f"[{n}] {_snippet_head(s)}\n{text[:per_snippet_chars]}"
        if used + len(block) > total_chars:
            lines.append(f"...（证据过长，已截断，共 {len(all_snippets)} 条）")
            break
        lines.append(block)
        used += len(block)

    # ③ 追加综合稿（synthesize 产出）
    for tool_name, ans_text in answers:
        lines.append(f"[{tool_name} 综合稿] {ans_text[:per_snippet_chars]}")

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


def _build_history_block(session_summary: str, session_raw: str) -> str:
    """组装多轮历史注入块（Phase 3）：Layer 2 摘要 + Layer 1 原文。

    两者都只帮解析指代（「那第二款呢？」），模板里明确声明不是证据，不污染
    grounding 判分。注入总量受 HISTORY_SUMMARY_MAX_CHARS 约束（超出则截断），
    避免挤占证据预算。无任何历史时返回空串。
    """
    if not session_summary and not session_raw:
        return ""
    parts: list[str] = []
    if session_summary:
        parts.append(HISTORY_SUMMARY_SECTION.format(summary=session_summary))
    if session_raw:
        parts.append(f"[最近对话原文（仅供理解上下文与指代，不是证据）]\n{session_raw}")
    block = "\n\n".join(parts)
    if len(block) > HISTORY_SUMMARY_MAX_CHARS:
        block = block[:HISTORY_SUMMARY_MAX_CHARS]
    return block + "\n\n"


def _generate_draft(objective: str, observations: list[dict], errors: list[dict],
                    *, session_summary: str = "", session_raw: str = "") -> str:
    """Generate a draft answer using LLM with tool observations as context.

    ``session_summary`` / ``session_raw`` 非空时（多轮对话）注入 user prompt 前缀：
    它们只帮解析指代（「那第二款呢？」），模板里明确声明不是证据，不污染
    grounding 判分（见 prompts.HISTORY_SUMMARY_SECTION 与 _build_history_block）。
    """
    context = _format_evidence_block(observations)
    error_text = json.dumps(errors[-5:], ensure_ascii=False)[:300] if errors else "无错误"
    history_block = _build_history_block(session_summary, session_raw)

    try:
        from src.llm import chat_completion
        system = GENERATOR_SYSTEM_PROMPT
        user = (
            f"{history_block}"
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
        obs_text = _format_evidence_block(
            observations, per_snippet_chars=500, total_chars=EVALUATOR_EVIDENCE_TOTAL_CHARS
        )

        system = EVALUATOR_SYSTEM_PROMPT
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
