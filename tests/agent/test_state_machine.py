"""Harness V2 Phase 4 测试：状态机转移表完整性 / BudgetTracker / 审计落盘 / prompt 版本哈希 / 幂等续跑。

与 tests/agent/test_agent_harness.py 的分工：那份文件覆盖 Phase 1-3 的行为（护栏闭环、
HITL、CHECKPOINT 记录本身）；本文件专门覆盖 Phase 4 新增的可观测性能力，以及补充
一条 Phase 2 幂等续跑的回归测试（防止 Phase 4 改动 _run_plan_step 时破坏续跑语义）。
"""
from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

from src.agent.harness import (
    DEFAULT_BUDGET,
    HarnessContext,
    _apply_caller_budget,
    _emit_audit,
    _persist_audit_events,
    _prompt_version_hash,
    _record_checkpoint,
    _should_skip_step,
    _STATE_HANDLERS,
)
from src.agent.state_machine import (
    BudgetTracker,
    HarnessState,
    RouteBudget,
    TERMINAL_STATES,
)


# ── 测试辅助（与 test_agent_harness.py 中同名辅助等价，但独立实现，避免跨测试模块导入）──

def _make_ctx(**overrides: Any) -> HarnessContext:
    """构造用于直接驱动 handler 的 HarnessContext（run 用 MagicMock，不落盘）。"""
    kwargs: dict[str, Any] = dict(
        run_id="test-run",
        objective="劳动合同试用期最长多久",
        tenant_id="t_test",
        user_id="u1",
        user_context={"roles": ["researcher"]},
        session_id=None,
        ticket_id=None,
        t_start=time.perf_counter(),
        budget=dict(DEFAULT_BUDGET),
        run=MagicMock(),
    )
    kwargs.update(overrides)
    ctx = HarnessContext(**kwargs)
    _apply_caller_budget(ctx)
    return ctx


class _SeqRegistry:
    """假工具注册表：local_search 每次返回全新 citation_id（规避循环检测），并记录调用。"""

    def __init__(self, tools: dict | None = None) -> None:
        self.calls: list[dict] = []
        self._n = 0
        self._tools = dict(tools or {})

    def get(self, name: str):
        return self._tools.get(name)

    async def execute(self, name, params, *, user_context=None, tenant_id=None):
        from src.agent.tool_registry import ToolCallResult
        self.calls.append({"tool": name, "params": dict(params)})
        self._n += 1
        if name == "local_search":
            return ToolCallResult(name, True, data={
                "query": params.get("query"), "count": 1,
                "snippets": [{"citation_id": f"CIT-{self._n}",
                              "text": f"劳动合同法第{self._n}条：试用期……",
                              "file_name": "law.md", "heading": "h"}],
            })
        return ToolCallResult(name, True, data={"ok": True, "n": self._n})


async def _drive_with_audit(ctx: HarnessContext, start: HarnessState,
                            stop_states: set[HarnessState], max_iter: int = 60) -> HarnessState:
    """从 start 驱动状态机，每次转移后调用 _emit_audit（模拟真实 while 循环），
    直到进入 stop_states 之一（或超过 max_iter 防失控）。"""
    state = start
    n = 0
    while state not in stop_states and n < max_iter:
        n += 1
        prev = state
        state = await _STATE_HANDLERS[state](ctx)
        _emit_audit(ctx, prev, state)
    return state


# ── 1. 转移表完整性 ──────────────────────────────────────────────────────

def test_state_handlers_cover_all_non_terminal_states():
    """_STATE_HANDLERS 必须覆盖除终态（DONE/REJECTED）外的所有 HarnessState。"""
    expected = set(HarnessState) - TERMINAL_STATES
    assert set(_STATE_HANDLERS.keys()) == expected, (
        f"缺失或多余的状态 handler: 缺失={expected - set(_STATE_HANDLERS)}, "
        f"多余={set(_STATE_HANDLERS) - expected}"
    )


def test_terminal_states_have_no_handler():
    """终态（DONE/REJECTED）不应有 handler（while 循环遇到即退出，不会去查表）。"""
    for state in TERMINAL_STATES:
        assert state not in _STATE_HANDLERS


# ── 2. BudgetTracker 单元测试 ─────────────────────────────────────────────

def _budget(**overrides: Any) -> RouteBudget:
    b: RouteBudget = {
        "max_steps": 10,
        "max_tool_calls": 20,
        "max_tokens": 32000,
        "max_rewrite_retries": 2,
        "max_latency_ms": 120000,
        "max_transitions": 30,
        "loop_detect_threshold": 2,
    }
    b.update(overrides)  # type: ignore[typeddict-item]
    return b


def test_budget_tracker_snapshot_format():
    """snapshot() 应返回 6 个维度、格式均为 "已用/上限" 的字符串。"""
    tracker = BudgetTracker(budget=_budget())
    tracker.steps_used = 3
    tracker.tool_calls_used = 5
    tracker.tokens_estimated = 1000
    tracker.rewrite_retries_used = 1
    tracker.latency_ms = 2500
    tracker.transitions = 7
    snap = tracker.snapshot()
    assert snap == {
        "steps": "3/10",
        "tool_calls": "5/20",
        "tokens_est": "1000/32000",
        "rewrites": "1/2",
        "latency_ms": "2500/120000",
        "transitions": "7/30",
    }


def test_budget_tracker_not_over_budget_when_under_all_limits():
    tracker = BudgetTracker(budget=_budget())
    tracker.steps_used = 10       # 等于上限，不算超（严格大于才超）
    tracker.tool_calls_used = 20
    tracker.tokens_estimated = 32000
    tracker.latency_ms = 120000
    tracker.transitions = 30
    assert tracker.is_over_budget() is False


def test_budget_tracker_over_budget_each_dimension():
    """任一维度超限都应让 is_over_budget() 返回 True。"""
    for field_name, over_value in [
        ("steps_used", 11),
        ("tool_calls_used", 21),
        ("tokens_estimated", 32001),
        ("latency_ms", 120001),
        ("transitions", 31),
    ]:
        tracker = BudgetTracker(budget=_budget())
        setattr(tracker, field_name, over_value)
        assert tracker.is_over_budget() is True, f"{field_name}={over_value} 应判定超预算"


# ── 3. 预算超转：状态机在超限时进入 DEGRADED ────────────────────────────────

async def test_execute_loop_degrades_when_transitions_over_budget():
    """max_transitions=3 的极小预算下，多步计划跑到超限时应转入 DEGRADED（而非 DRAFT）。"""
    reg = _SeqRegistry()
    # 旧的 per-step 护栏（ctx.max_*）全部调高，确保只有 BudgetTracker.transitions 触发降级
    ctx = _make_ctx(registry=reg, budget={
        "max_steps": 100, "max_tool_calls": 100, "max_latency_ms": 100_000,
        "step_timeout_seconds": 30.0, "max_transitions": 100,
        "max_rewrite_attempts": 5, "loop_detect_threshold": 100,
    })
    ctx.budget_tracker.budget = _budget(max_transitions=3)
    ctx.plan = [{"type": "retrieve", "query": f"Q{i}"} for i in range(4)]

    final = await _drive_with_audit(
        ctx, HarnessState.EXECUTE_LOOP,
        {HarnessState.DEGRADED, HarnessState.DRAFT},
    )

    assert final == HarnessState.DEGRADED, f"超预算应转 DEGRADED，实际={final}"
    assert ctx.degraded_forced is True
    assert ctx.budget_tracker.transitions > 3
    assert any(a.get("step") == "budget_exceeded" for a in ctx.audit), "审计应记录 budget_exceeded"
    assert any(e.get("error") == "budget_exceeded" for e in ctx.errors), "errors 应记录 budget_exceeded"
    # 触发降级对应的那条状态转移审计事件里应带上 error_class/error_action
    degraded_events = [e for e in ctx.state_events if e["to_state"] == HarnessState.DEGRADED.value]
    assert degraded_events, "应有一条转入 DEGRADED 的审计事件"
    meta = degraded_events[0]["metadata"]
    assert meta.get("error_class") == "budget_exceeded"
    assert meta.get("error_action") == "degrade"
    assert "budget_consumed" in meta


async def test_execute_loop_does_not_degrade_when_under_budget():
    """预算充足时应正常跑完计划到 DRAFT，不触发降级。"""
    reg = _SeqRegistry()
    ctx = _make_ctx(registry=reg, budget={
        "max_steps": 100, "max_tool_calls": 100, "max_latency_ms": 100_000,
        "step_timeout_seconds": 30.0, "max_transitions": 100,
        "max_rewrite_attempts": 5, "loop_detect_threshold": 100,
    })
    ctx.budget_tracker.budget = _budget(max_transitions=100)
    ctx.plan = [{"type": "retrieve", "query": f"Q{i}"} for i in range(3)]

    final = await _drive_with_audit(
        ctx, HarnessState.EXECUTE_LOOP,
        {HarnessState.DEGRADED, HarnessState.DRAFT},
    )

    assert final == HarnessState.DRAFT, f"预算充足时应正常跑到 DRAFT，实际={final}"
    assert ctx.degraded_forced is False
    assert ctx.budget_tracker.steps_used == 3
    assert ctx.budget_tracker.tool_calls_used == 3


# ── 4. 审计事件落盘（JSONL）────────────────────────────────────────────────

def test_persist_audit_events_writes_jsonl(tmp_path, monkeypatch):
    """_persist_audit_events 应按 run_id 写入 <TRACE_DIR>/<日期>/<run_id>.audit.jsonl。"""
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    ctx = _make_ctx(run_id="run-abc-123")
    ctx.state_events.append({
        "run_id": "run-abc-123", "from_state": "init", "to_state": "security_check",
        "timestamp": 1234.5, "metadata": {"budget_consumed": {"steps": "0/10"}},
    })
    ctx.state_events.append({
        "run_id": "run-abc-123", "from_state": "security_check", "to_state": "context_load",
        "timestamp": 1235.0, "metadata": {},
    })

    _persist_audit_events(ctx)

    from datetime import date
    day_dir = tmp_path / date.today().isoformat()
    files = list(day_dir.glob("*.audit.jsonl"))
    assert len(files) == 1, f"应恰好落一个审计文件，实际={files}"
    assert files[0].name == "run-abc-123.audit.jsonl"

    lines = files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["to_state"] == "security_check"
    assert json.loads(lines[1])["to_state"] == "context_load"


def test_persist_audit_events_is_incremental(tmp_path, monkeypatch):
    """重复调用不应重复写入已经落盘的事件（增量 flush）。"""
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    ctx = _make_ctx(run_id="run-inc")
    ctx.state_events.append({"run_id": "run-inc", "from_state": "a", "to_state": "b",
                             "timestamp": 1.0, "metadata": {}})
    _persist_audit_events(ctx)

    ctx.state_events.append({"run_id": "run-inc", "from_state": "b", "to_state": "c",
                             "timestamp": 2.0, "metadata": {}})
    _persist_audit_events(ctx)

    from datetime import date
    filepath = tmp_path / date.today().isoformat() / "run-inc.audit.jsonl"
    lines = filepath.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2, f"两次调用应共写入 2 条不重复事件，实际={len(lines)}"
    assert json.loads(lines[0])["to_state"] == "b"
    assert json.loads(lines[1])["to_state"] == "c"


def test_persist_audit_events_failure_does_not_raise(tmp_path, monkeypatch):
    """落盘失败（如目录不可写）不应抛出异常，只记日志。"""
    # 指向一个文件而非目录，mkdir 会失败
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv("TRACE_DIR", str(blocker))
    ctx = _make_ctx(run_id="run-fail")
    ctx.state_events.append({"run_id": "run-fail", "from_state": "a", "to_state": "b",
                             "timestamp": 1.0, "metadata": {}})
    # 不应抛出
    _persist_audit_events(ctx)


def test_safe_run_id_in_filename(tmp_path, monkeypatch):
    """run_id 中的非法文件名字符应被替换为下划线。"""
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    ctx = _make_ctx(run_id="run/with:bad*chars")
    ctx.state_events.append({"run_id": "run/with:bad*chars", "from_state": "a",
                             "to_state": "b", "timestamp": 1.0, "metadata": {}})
    _persist_audit_events(ctx)
    from datetime import date
    day_dir = tmp_path / date.today().isoformat()
    files = list(day_dir.glob("*.audit.jsonl"))
    assert len(files) == 1
    assert "/" not in files[0].name and ":" not in files[0].name and "*" not in files[0].name


# ── 5. prompt 版本哈希 ─────────────────────────────────────────────────────

def test_prompt_version_hash_is_stable_and_12_chars():
    h1 = _prompt_version_hash()
    h2 = _prompt_version_hash()
    assert h1 == h2, "同内容多次调用应得到相同哈希"
    assert len(h1) == 12
    assert all(c in "0123456789abcdef" for c in h1), "应为 sha256 十六进制前缀"


def test_prompt_version_hash_changes_when_prompt_changes(monkeypatch):
    """篡改任一 prompt 常量后，哈希应变化（证明哈希确实覆盖了这些 prompt）。"""
    import src.agent.harness as harness_mod
    original = _prompt_version_hash()
    monkeypatch.setattr(harness_mod, "PLANNER_SYSTEM_PROMPT",
                        harness_mod.PLANNER_SYSTEM_PROMPT + "（篡改）")
    assert _prompt_version_hash() != original


# ── 6. 幂等恢复（补充 Phase 2）──────────────────────────────────────────────

def test_should_skip_step_after_preset_checkpoint():
    """预设断点后，_should_skip_step 应对同 index+同参数返回 True，其余返回 False。"""
    ctx = _make_ctx()
    params = {"query": "Q1", "top_k": 5}
    _record_checkpoint(ctx, 0, "local_search", params, {"snippets": [{"citation_id": "C0"}]})

    assert _should_skip_step(ctx, 0, "local_search", params) is True
    assert _should_skip_step(ctx, 1, "local_search", params) is False        # 不同 index
    assert _should_skip_step(ctx, 0, "local_search",
                             {"query": "Q2", "top_k": 5}) is False          # 不同参数
    assert _should_skip_step(ctx, 0, "other_tool", params) is False         # 不同工具名（参数指纹含工具名）


async def test_preset_checkpoint_skips_step_on_resume():
    """续跑场景：已完成的步骤不再重复调用工具，仅执行未完成步骤。"""
    reg = _SeqRegistry()
    ctx = _make_ctx(registry=reg)
    ctx.plan = [{"type": "retrieve", "query": "Q1"}, {"type": "retrieve", "query": "Q2"}]
    # 模拟中断前 step0 已成功执行并记录断点
    _record_checkpoint(ctx, 0, "local_search", {"query": "Q1", "top_k": 5},
                       {"snippets": [{"citation_id": "OLD-0"}]})
    reg.calls.clear()

    final = await _drive_with_audit(ctx, HarnessState.EXECUTE_LOOP, {HarnessState.DRAFT})

    assert final == HarnessState.DRAFT
    assert len(reg.calls) == 1, f"续跑应只执行未完成的步骤，实际调用={reg.calls}"
    assert reg.calls[0]["params"]["query"] == "Q2"
    assert any(a.get("step") == "checkpoint_skip" and a.get("index") == 0 for a in ctx.audit)
    # step0 的断点没有被重复记录（仍只有预设那一条）
    assert len([cp for cp in ctx.checkpoints if cp["step_index"] == 0]) == 1
    assert len(ctx.checkpoints) == 2


# ── 7.（可选能力）metrics 导出 ─────────────────────────────────────────────

def test_export_metrics_writes_jsonl(tmp_path, monkeypatch):
    from src.agent.harness import _export_metrics
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    ctx = _make_ctx(run_id="run-metrics")
    ctx.risk_level = "low"
    ctx.eval_result = {"passed": True, "issues": []}
    ctx.needs_human = False

    _export_metrics(ctx)

    from datetime import date
    filepath = tmp_path / date.today().isoformat() / "metrics.jsonl"
    assert filepath.exists(), "应写入 metrics.jsonl"
    lines = filepath.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["run_id"] == "run-metrics"
    assert row["risk_level"] == "low"
    assert row["grounded"] is True
    assert row["needs_human"] is False
    assert len(row["prompt_version"]) == 12


# ── 8. 端到端：完整 run 的审计字段覆盖（4.1 标准化字段表的回归锁）───────

#: 4.1 表格里“一次 FULL_PIPELINE run 必然会触发”的标准化字段。
#: error_class / error_action 只在错误路径出现，由
#: test_execute_loop_degrades_when_transitions_over_budget 单独覆盖，不在此列。
_EXPECTED_AUDIT_FIELDS = {
    "budget_consumed",        # 每条转移都带（_emit_audit 自动附加）
    "session_id", "context_layers_used",                       # CONTEXT_LOAD
    "risk_level", "intent", "route_decision",                  # RISK_INTENT
    "tool_name", "tool_args_hash",                             # 工具调用前
    "tool_result_status", "tool_latency_ms",                   # 工具调用后
    "step_index", "result_hash",                               # CHECKPOINT
    "guard_result", "guard_reason",                            # 输出护栏
    "eval_grounded", "eval_issues", "rewrite_count",           # 评估完成
    "hitl_type", "hitl_decision",                              # HITL
    "prompt_version",                                          # PERSIST（最终）
}


async def test_full_run_audit_jsonl_field_coverage(tmp_path, monkeypatch):
    """跑一次完整 run_agent_harness，验证审计 JSONL 落盘且 4.1 标准化字段全部出现。

    这是 Phase 4 可观测性的端到端回归锁：某个 handler 漏填 set_audit_meta、
    或工具失败路径忘了写 tool_latency_ms，都会在这里被捕获（单元测试只能
    证明单个 handler 对，证明不了整条链路拼接后字段仍齐）。
    """
    from unittest.mock import patch

    from src.agent.harness import run_agent_harness

    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))

    # 离线：LLM 返回空串（→ rule-based 规划 + 模板兜底），向量索引不可用
    # （→ local_search 失败，正好走工具失败路径的审计字段）。
    with patch("src.llm.chat_completion", return_value=""), \
         patch("src.vector_index.get_vector_index", return_value=None):
        result = await run_agent_harness(
            objective="劳动合同试用期最长多久",
            tenant_id="t_test",
            user_id="u1",
            user_context={"roles": ["researcher"]},
        )

    assert result.run_id
    assert result.status in ("completed", "waiting_approval", "failed")

    from datetime import date
    day_dir = tmp_path / date.today().isoformat()
    audit_file = day_dir / f"{result.run_id}.audit.jsonl"
    assert audit_file.exists(), f"应落盘 {audit_file.name}，实际目录内容={list(day_dir.iterdir())}"

    events = [json.loads(line) for line in
              audit_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
    assert len(events) >= 10, f"完整 run 应有多条转移事件，实际={len(events)}"

    # 每条事件的结构契约
    for e in events:
        assert set(e.keys()) == {"run_id", "from_state", "to_state", "timestamp", "metadata"}
        assert e["run_id"] == result.run_id
        assert isinstance(e["metadata"], dict)
        assert "budget_consumed" in e["metadata"], "每条转移都应带预算快照"

    # 首尾转移正确（INIT 开始、DONE/REJECTED 结束）
    assert events[0]["from_state"] == HarnessState.INIT.value
    assert events[-1]["to_state"] in {s.value for s in TERMINAL_STATES}

    # 4.1 标准化字段覆盖
    union: set[str] = set()
    for e in events:
        union |= set(e["metadata"].keys())
    missing = _EXPECTED_AUDIT_FIELDS - union
    assert not missing, f"审计元数据缺少 4.1 要求的字段: {sorted(missing)}"

    # 工具调用转移：只要带了 tool_name，就必须同时带 status 与 latency
    for e in events:
        meta = e["metadata"]
        if "tool_name" in meta and "tool_result_status" in meta:
            assert "tool_latency_ms" in meta, (
                f"{e['from_state']}->{e['to_state']} 调了工具但缺 tool_latency_ms: {sorted(meta)}"
            )
            assert isinstance(meta["tool_latency_ms"], int)

    # 最后一条（PERSIST→DONE）应带 prompt_version 与最终预算快照
    last_meta = events[-1]["metadata"]
    if events[-1]["to_state"] == HarnessState.DONE.value:
        assert len(last_meta.get("prompt_version", "")) == 12
    snap = last_meta["budget_consumed"]
    assert set(snap.keys()) == {"steps", "tool_calls", "tokens_est",
                                "rewrites", "latency_ms", "transitions"}

    # metrics.jsonl 恰好一行（每次 run 一条）
    metrics_file = day_dir / "metrics.jsonl"
    assert metrics_file.exists(), "应导出 metrics.jsonl"
    mrows = [json.loads(line) for line in
             metrics_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
    assert len(mrows) == 1, f"一次 run 应恰好一条 metrics，实际={len(mrows)}"
    m = mrows[0]
    assert m["run_id"] == result.run_id
    assert len(m["prompt_version"]) == 12
    assert m["total_latency_ms"] >= 0
    for key in ("route", "risk_level", "tool_calls", "rewrite_count",
                "grounded", "needs_human", "timestamp"):
        assert key in m, f"metrics 缺少字段 {key}"
