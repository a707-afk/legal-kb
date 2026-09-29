"""JSONL 落盘引擎（替代旧 src.db.engine 的 SQLAlchemy async session）。

保留旧调用形态 `async with get_sessionmaker()() as session: session.add(obj); await session.commit()`，
但底层是「一个 run 一个文件、逐条追加」的 JSONL（BLUEPRINT D-12 附）。
所有写操作 fire-and-forget：失败只记日志、绝不抛，避免拖垮业务流程。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def runs_dir() -> Path:
    """Agent run JSONL 的根目录（可用 AGENT_RUN_DIR 覆盖）。"""
    d = Path(os.getenv("AGENT_RUN_DIR", "data/agent_runs"))
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - 只读文件系统等
        logger.exception("无法创建 runs 目录: %s", d)
    return d


def run_file(run_id: str) -> Path:
    """按 run_id 分文件：无写竞争，可单独 diff / 回放。"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (run_id or "unknown"))
    return runs_dir() / f"{safe}.jsonl"


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """追加一行 JSON；失败只记日志不抛。"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("append_jsonl 失败: %s", path)


class JsonlSession:
    """模仿 SQLAlchemy async session 的最小 JSONL 会话。

    收集 add() 的对象，commit() 时按 run 落盘：AgentRun 写首行，AgentStep 追加。
    """

    def __init__(self) -> None:
        self._pending: list[Any] = []

    async def __aenter__(self) -> "JsonlSession":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def add(self, obj: Any) -> None:
        self._pending.append(obj)

    async def commit(self) -> None:
        pending, self._pending = self._pending, []
        # 先写 run 头，再写 steps，保证回放时首行是 run
        pending.sort(key=lambda o: 0 if getattr(o, "objective", None) is not None and hasattr(o, "budget_json") else 1)
        for obj in pending:
            to_dict = getattr(obj, "to_dict", None)
            if to_dict is None:
                continue
            run_id = getattr(obj, "run_id", None) or getattr(obj, "id", None)
            append_jsonl(run_file(run_id), to_dict())


def get_sessionmaker():
    """返回会话工厂；`get_sessionmaker()()` 得到异步上下文会话（兼容旧调用形态）。"""
    return JsonlSession
