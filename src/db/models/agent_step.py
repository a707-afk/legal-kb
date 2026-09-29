"""AgentStep：Agent 运行的单步记录（JSONL dataclass，非 ORM），用于审计回放。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentStep:
    """一步 plan/execute/observe/evaluate/approve 记录，追加到所属 run 的 JSONL。"""

    id: str
    run_id: str
    tenant_id: str = "default"
    step_index: int = 0
    step_type: str = ""              # plan | execute | observe | evaluate | approve
    input_json: str | None = None
    output_json: str | None = None
    tool_name: str | None = None
    tool_params_json: str | None = None
    tool_result_json: str | None = None
    permission_check: str | None = None
    latency_ms: float = 0.0
    error_message: str | None = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_type"] = "step"
        return d
