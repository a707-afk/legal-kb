"""结构化 JSON 行日志（阶段 F 最小切片，无 Langfuse/OTel）。"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_structured_event(
    trace_id: str | None,
    event: str,
    **fields: Any,
) -> None:
    """一行 JSON：trace_id + event + 业务字段。"""
    row: dict[str, Any] = {"trace_id": trace_id, "event": event}
    row.update(fields)
    logger.info("%s", json.dumps(row, ensure_ascii=False))
