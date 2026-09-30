"""Harness V2 状态机类型定义（Phase 1）。

本模块是**纯类型/常量**模块，不依赖 LLM、runtime 或 harness 实现，
供 ``src/agent/harness.py`` 的状态机驱动循环使用：

- ``HarnessState``     ：状态机所有节点（主流程 / 路由分支 / HITL / 错误处理 / 终态）
- ``ErrorAction``      ：错误分类后的处置动作
- ``RouteBudget``      ：按路由档位约束的预算（步数/工具调用/token/改写/时延/转移/循环阈值）
- ``RouteDecision``    ：RISK_INTENT 阶段产出的路由决策（不可变 NamedTuple）
- ``AuditMetadata`` / ``AuditEvent``：状态转移审计载荷
- ``Checkpoint``       ：执行循环的断点续跑快照
- ``UserContext``      ：调用方身份上下文
- ``TERMINAL_STATES``  ：终态集合（while 循环退出条件）

设计取舍见 why/06-为什么自研Harness不用LangChain.md：状态机是**显式**的，
每个节点只做一件事并返回下一个状态，控制流可读、可回放、可审计。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, NamedTuple, TypedDict


class HarnessState(Enum):
    """状态机所有状态节点。"""

    # ── 主流程 ──
    INIT = "init"
    SECURITY_CHECK = "security_check"
    CONTEXT_LOAD = "context_load"
    RISK_INTENT = "risk_intent"
    ROUTE = "route"
    # ── 路由分支 ──
    FAST_PATH = "fast_path"
    PLAN = "plan"
    PLAN_VALIDATE = "plan_validate"
    EXECUTE_LOOP = "execute_loop"
    DRAFT = "draft"
    OUTPUT_GUARD = "output_guard"
    EVALUATE = "evaluate"
    REWRITE = "rewrite"
    # ── HITL ──
    HITL_PRE = "hitl_pre"
    HITL_ACTION = "hitl_action"
    HITL_OUTPUT = "hitl_output"
    # ── 错误处理 ──
    ERROR_CLASSIFY = "error_classify"
    RETRY_BACKOFF = "retry_backoff"
    DEGRADED = "degraded"
    # ── 终态 / 落盘 ──
    CHECKPOINT = "checkpoint"
    PERSIST = "persist"
    DONE = "done"
    REJECTED = "rejected"


class ErrorAction(Enum):
    """错误分类后的处置动作。"""

    RETRY = "retry"
    DEGRADE = "degrade"
    ABORT = "abort"
    SKIP = "skip"


class RouteBudget(TypedDict):
    """按路由档位约束的预算。"""

    max_steps: int
    max_tool_calls: int
    max_tokens: int
    max_rewrite_retries: int
    max_latency_ms: int
    max_transitions: int
    loop_detect_threshold: int


BUDGET_FAST: RouteBudget = {
    "max_steps": 3,
    "max_tool_calls": 3,
    "max_tokens": 4000,
    "max_rewrite_retries": 0,
    "max_latency_ms": 30000,
    "max_transitions": 10,
    "loop_detect_threshold": 2,
}
BUDGET_FULL: RouteBudget = {
    "max_steps": 10,
    "max_tool_calls": 20,
    "max_tokens": 32000,
    "max_rewrite_retries": 2,
    "max_latency_ms": 120000,
    "max_transitions": 30,
    "loop_detect_threshold": 2,
}
BUDGET_HITL: RouteBudget = {
    "max_steps": 10,
    "max_tool_calls": 20,
    "max_tokens": 32000,
    "max_rewrite_retries": 1,
    "max_latency_ms": 300000,
    "max_transitions": 40,
    "loop_detect_threshold": 2,
}


@dataclass
class BudgetTracker:
    """运行时预算消耗追踪器（Phase 4）。

    与 ``RouteBudget`` 的**静态上限**相对，本类记录一次 run 的**实际消耗**，供：
      - ``snapshot()``      ：写入每条状态转移审计事件的 ``budget_consumed`` 字段（可观测）
      - ``is_over_budget()``：``_handle_execute_loop`` 的软护栏，任一维度超限即转 DEGRADED

    ``transitions`` 计的是**状态机级**转移次数（每次 ``_emit_audit`` +1），语义上独立于
    harness 里旧的 per-step ``transition_count``（那个只数计划步的执行次数）——两者是
    并行的两层护栏，不共用计数器。
    """

    budget: RouteBudget
    steps_used: int = 0
    tool_calls_used: int = 0
    tokens_estimated: int = 0
    rewrite_retries_used: int = 0
    latency_ms: int = 0
    transitions: int = 0

    def snapshot(self) -> dict:
        """当前消耗 / 上限的可读快照（写入审计 metadata 的 ``budget_consumed``）。"""
        return {
            "steps": f"{self.steps_used}/{self.budget['max_steps']}",
            "tool_calls": f"{self.tool_calls_used}/{self.budget['max_tool_calls']}",
            "tokens_est": f"{self.tokens_estimated}/{self.budget['max_tokens']}",
            "rewrites": f"{self.rewrite_retries_used}/{self.budget['max_rewrite_retries']}",
            "latency_ms": f"{self.latency_ms}/{self.budget['max_latency_ms']}",
            "transitions": f"{self.transitions}/{self.budget['max_transitions']}",
        }

    def is_over_budget(self) -> bool:
        """任一维度超预算即返回 True。"""
        return (
            self.steps_used > self.budget["max_steps"]
            or self.tool_calls_used > self.budget["max_tool_calls"]
            or self.tokens_estimated > self.budget["max_tokens"]
            or self.latency_ms > self.budget["max_latency_ms"]
            or self.transitions > self.budget["max_transitions"]
        )


class RouteDecision(NamedTuple):
    """RISK_INTENT 阶段的路由决策（不可变）。"""

    route: Literal["fast", "full", "hitl_pre", "reject"]
    risk_level: str
    intent: str
    budget: RouteBudget


class AuditMetadata(TypedDict, total=False):
    """单次状态转移携带的审计元数据（全部可选）。"""

    route_decision: dict | None
    risk_level: str
    intent: str
    budget_consumed: dict
    budget_remaining: dict
    tool_name: str
    tool_args_hash: str
    tool_result_status: str
    tool_latency_ms: int
    guard_result: str
    guard_reason: str
    eval_grounded: bool
    eval_issues: list[str]
    rewrite_count: int
    hitl_type: str
    hitl_decision: str
    error_class: str
    error_action: str
    session_id: str
    context_layers_used: list[str]
    step_index: int
    result_hash: str | None
    prompt_version: str


class AuditEvent(TypedDict):
    """一条状态转移审计事件。"""

    run_id: str
    from_state: str
    to_state: str
    timestamp: float
    metadata: AuditMetadata


class Checkpoint(TypedDict):
    """执行循环断点快照（Phase 1 内存态，Phase 4 落盘续跑）。"""

    run_id: str
    step_index: int
    tool_name: str
    tool_args_hash: str
    result_hash: str | None
    state: str
    timestamp: float


class UserContext(TypedDict, total=False):
    """调用方身份上下文。"""

    user_id: str
    roles: list[str]
    scopes: list[str]
    tenant_id: str


#: 终态集合：状态机 while 循环遇到终态即退出。
TERMINAL_STATES = frozenset({HarnessState.DONE, HarnessState.REJECTED})

__all__ = [
    "HarnessState",
    "ErrorAction",
    "RouteBudget",
    "BUDGET_FAST",
    "BUDGET_FULL",
    "BUDGET_HITL",
    "BudgetTracker",
    "RouteDecision",
    "AuditMetadata",
    "AuditEvent",
    "Checkpoint",
    "UserContext",
    "TERMINAL_STATES",
]
