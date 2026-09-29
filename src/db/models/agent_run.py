"""AgentRun：一次 Agent 运行的头部记录（JSONL dataclass，非 ORM）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentRun:
    """一次 Agent 运行的元数据与最终结果。

    字段沿用旧 src.db.models.agent_run 的命名，落盘时序列化为 JSONL 首行。
    """

    id: str
    tenant_id: str = "default"
    user_id: str | None = None
    session_id: str | None = None
    ticket_id: str | None = None
    objective: str = ""
    user_query: str = ""
    status: str = "running"          # running | completed | waiting_approval | failed | terminated
    risk_level: str = "low"
    budget_json: str | None = None
    plan_json: str | None = None
    final_answer: str | None = None
    final_action: str | None = None
    human_review_required: bool = False
    total_steps: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    tool_error_count: int = 0
    permission_deny_count: int = 0
    errors_json: str | None = None
    approvals_json: str | None = None
    audit_trace_json: str | None = None
    termination_reason: str | None = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_type"] = "run"
        return d
