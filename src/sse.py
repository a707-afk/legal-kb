"""Server-Sent Events 辅助（text/event-stream）。"""
from __future__ import annotations

import json
from typing import Any, Iterator


def format_sse_event(event: str, data: Any) -> str:
    """格式化为 SSE 单行 data（JSON）。"""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def chunk_text(text: str, chunk_size: int = 12) -> Iterator[str]:
    """将完整文本分块，用于 LLM 无真流式时的模拟输出。"""
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
