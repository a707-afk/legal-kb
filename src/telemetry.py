"""轻量链路追踪（span）—— D-15「检索链路 trace + 回放」的接入点。

当前只做 span 的 enter/exit 计时并落一条结构化日志（复用 logging_utils）。
不引入 OpenTelemetry（BLUEPRINT 第五章：OTel/Prometheus 明确不做）。
D2.7 的完整 JSONL trace + replay_trace.py 会在此基础上扩展。
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from src.logging_utils import log_structured_event

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, *, trace_id: str | None = None, **attrs: Any) -> Iterator[None]:
    """同步上下文管理器：记录一个 span 的名称、耗时与附加属性。"""
    t0 = time.perf_counter()
    error: str | None = None
    try:
        yield
    except Exception as exc:  # 记录后原样抛出，不吞异常
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_structured_event(
            trace_id,
            "span",
            span=name,
            duration_ms=duration_ms,
            error=error,
            **attrs,
        )
