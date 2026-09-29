"""审计服务：把不可变审计记录追加到 JSONL（替代旧 policy_audit_logs / tool_call 表）。

决策依据 BLUEPRINT D-12 附 / D-15：个人项目不引入数据库，审计以 JSONL 落盘，
可直接 diff 与回放。所有写入 fire-and-forget：失败只记日志、不抛，避免拖垮业务流程。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db.engine import append_jsonl

logger = logging.getLogger(__name__)


def _audit_file(name: str) -> Path:
    """审计 JSONL 路径（可用 AUDIT_LOG_DIR 覆盖）。"""
    return Path(os.getenv("AUDIT_LOG_DIR", "data/audit")) / f"{name}.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def write_audit_log(
    *,
    tenant_id: str,
    event_type: str,
    action: str,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: dict[str, Any] | None = None,
    risk_level: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """写一条审计日志到 JSONL（fire-and-forget：失败只记日志，不抛）。"""
    try:
        append_jsonl(_audit_file("policy_audit_log"), {
            "_type": "audit_log",
            "ts": _now(),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "event_type": event_type,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "detail": detail,
            "risk_level": risk_level,
            "ip_address": ip_address,
            "user_agent": user_agent,
        })
    except Exception:
        logger.exception("Failed to write audit log: event=%s action=%s", event_type, action)


async def write_tool_call_audit(
    *,
    tenant_id: str,
    tool_name: str,
    params: dict | None = None,
    result: dict | None = None,
    success: bool = True,
    error_message: str | None = None,
    permission_result: str | None = None,
    permission_reason: str | None = None,
    idempotency_key: str | None = None,
    latency_ms: float = 0.0,
    run_id: str | None = None,
) -> None:
    """写一条工具调用审计到 JSONL（fire-and-forget：失败只记日志，不抛）。"""
    try:
        append_jsonl(_audit_file("tool_call"), {
            "_type": "tool_call",
            "ts": _now(),
            "run_id": run_id,
            "tenant_id": tenant_id,
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "success": success,
            "error_message": error_message,
            "permission_result": permission_result,
            "permission_reason": permission_reason,
            "idempotency_key": idempotency_key,
            "latency_ms": latency_ms,
        })
    except Exception:
        logger.exception("Failed to write tool call audit: tool=%s", tool_name)
