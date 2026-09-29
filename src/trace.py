"""检索链路 trace：一次查询一条结构化 JSONL + 按 query_id 回放（D-15）。

为什么需要（而不是 Prometheus）
------------------------------
Prometheus 能看延迟和错误率，但 RAG 的四种失败在指标上**完全看不见**：
没召回 / 排序错 / 用错片段 / 引用错条号。能定位这四种的只有链路 trace。

落盘布局
--------
    data/traces/<YYYY-MM-DD>/<query_id>.jsonl

一个 query 一个文件（回放 O(1)，不用扫全量），一行一个 stage：
``{"ts","query_id","stage","latency_ms", ...业务字段}``。

只落候选的 file / score / status_code，**不落全文**（D-15 L3：避免泄露敏感信息）。
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "data/traces"


def traces_root() -> Path:
    d = Path(os.getenv("TRACE_DIR", _DEFAULT_DIR))
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover
        logger.exception("无法创建 trace 目录: %s", d)
    return d


def new_query_id(prefix: str = "q") -> str:
    """生成一个可读的 query_id：`q-20260928-153012-ab12cd`。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}-{uuid.uuid4().hex[:6]}"


def trace_path(query_id: str, *, day: str | None = None) -> Path:
    day = day or datetime.now().strftime("%Y-%m-%d")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in query_id)
    return traces_root() / day / f"{safe}.jsonl"


class TraceRecorder:
    """一次查询的 trace 记录器。``enabled=False`` 时全部方法都是 no-op。"""

    def __init__(self, query_id: str | None = None, *, enabled: bool = True) -> None:
        self.query_id = query_id or new_query_id()
        self.enabled = enabled
        self._t0 = time.perf_counter()
        self._path = trace_path(self.query_id) if enabled else None

    def record(self, stage: str, **fields: Any) -> None:
        """记录一个 stage。失败只记日志、绝不抛（不能因为 trace 拖垮检索）。"""
        if not self.enabled or self._path is None:
            return
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "query_id": self.query_id,
            "stage": stage,
            "elapsed_ms": round((time.perf_counter() - self._t0) * 1000, 2),
        }
        row.update(fields)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("写 trace 失败: %s", self._path)

    # ── 常用 stage 的便捷方法 ────────────────────────────────────

    @staticmethod
    def _brief(nodes: list[Any], limit: int = 5) -> list[dict]:
        """候选摘要：只留 file / score / status_code，不落全文。"""
        out: list[dict] = []
        for sn in nodes[:limit]:
            node = getattr(sn, "node", None)
            meta = dict(getattr(node, "metadata", None) or {})
            out.append({
                "file": Path(str(meta.get("file_path") or meta.get("file_name") or "")).name,
                "heading_path": str(meta.get("heading_path") or "")[:80],
                "status_code": meta.get("status_code"),
                "score": round(float(getattr(sn, "score", 0.0) or 0.0), 4),
            })
        return out

    def stage_candidates(self, stage: str, nodes: list[Any], **extra: Any) -> None:
        self.record(stage, count=len(nodes), top=self._brief(nodes), **extra)

    def finish(self, **fields: Any) -> None:
        self.record("finish", total_ms=round((time.perf_counter() - self._t0) * 1000, 2), **fields)


def create_trace(query_id: str | None = None) -> TraceRecorder:
    """按配置创建 recorder（`retrieval_trace_enabled=false` 时返回 no-op）。"""
    enabled = True
    try:
        from src.config_LEGACY_REFERENCE import get_settings

        enabled = bool(getattr(get_settings(), "retrieval_trace_enabled", True))
    except Exception:
        enabled = True
    return TraceRecorder(query_id, enabled=enabled)


def load_trace(query_id: str) -> list[dict]:
    """按 query_id 读出全部 stage（跨天搜索，返回按时间排序的记录）。"""
    rows: list[dict] = []
    root = traces_root()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in query_id)
    for p in sorted(root.glob(f"*/{safe}.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def list_recent_traces(limit: int = 20) -> list[tuple[str, Path]]:
    """列出最近的 trace（query_id, 文件），按修改时间倒序。"""
    root = traces_root()
    files = sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.stem, p) for p in files[:limit]]
