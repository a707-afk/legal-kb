"""Session memory manager: multi-turn context injection for Agent conversations.

两层结构：
- :class:`SessionMemory`：单个会话的消息列表与摘要/上下文构建（纯内存对象）。
- 模块级会话注册表（``get_or_create_session`` 等）：按 ``session_id`` 索引的
  进程内 LRU 存储，供 harness 在多轮对话中取回历史。与 ``src/cache.py`` 同风格：
  BLUEPRINT 第五章明确不做 Redis，进程内存储 + 有界淘汰即可。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Max context window: last N messages to inject as prompt history
MAX_CONTEXT_MESSAGES = 20
# Max total chars for injected context (to avoid blowing LLM context window)
MAX_CONTEXT_CHARS = 4000
# 每个会话最多保留的消息条数（有界，防止长会话内存无限增长）
MAX_MESSAGES_PER_SESSION = 60
# 注册表最多同时保留的会话数（LRU 淘汰最久未动的会话）
MAX_SESSIONS = 128


@dataclass
class SessionMemory:
    """分层会话记忆（Harness V2 Phase 3）。

    Layer 1: recent_raw     — 最近 N 轮原文（直接注入，零 LLM 成本）
    Layer 2: cached_summary — 第 N+1~M 轮的惰性摘要（按需生成，有缓存）
    Layer 3: 更早历史直接丢弃（受 MAX_MESSAGES_PER_SESSION 有界约束）

    Does NOT touch DB directly — works with Message-like dicts.
    The caller (service layer) is responsible for persistence.
    """

    # Layer 1 保留轮数（一轮 = user + assistant 各一条）；类常量，不做 dataclass 字段
    RECENT_TURNS = 3
    # Layer 2 摘要最大字符数
    SUMMARY_MAX_CHARS = 800

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 惰性摘要缓存 + 摘要已覆盖到的消息索引（增量更新的游标）
    _cached_summary: str = ""
    _summary_valid_until: int = 0

    def add_message(
        self,
        role: str,
        content: str,
        *,
        citations: dict | None = None,
        grounding: dict | None = None,
    ) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "citations": citations,
            "grounding": grounding,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        # 有界保留：超出上限时丢最旧的消息
        if len(self.messages) > MAX_MESSAGES_PER_SESSION:
            del self.messages[: len(self.messages) - MAX_MESSAGES_PER_SESSION]

    def build_context_prompt(self) -> str:
        """Build a condensed conversation history for LLM prompt injection."""
        if not self.messages:
            return ""

        recent = self.messages[-MAX_CONTEXT_MESSAGES:]
        lines: list[str] = ["--- 对话历史 ---"]
        total_chars = 0

        for m in reversed(recent):  # most recent first, truncated if too long
            role = m["role"]
            content = str(m.get("content", ""))
            prefix = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}.get(role, role)
            line = f"{prefix}: {content}"
            if total_chars + len(line) > MAX_CONTEXT_CHARS:
                break
            lines.append(line)
            total_chars += len(line)

        lines.reverse()
        return "\n".join(lines)

    # ── Harness V2 Phase 3: 分层上下文 API ────────────────────────────

    def get_recent_raw(self, n: int | None = None) -> str:
        """Layer 1: 取最近 N 轮原文，格式化为可直接注入的文本（零 LLM 成本）。"""
        n = n or self.RECENT_TURNS
        recent = self.messages[-(n * 2):]  # user+assistant 各一条为一轮
        if not recent:
            return ""
        parts: list[str] = []
        for m in recent:
            role_label = "用户" if m["role"] == "user" else "助手"
            content = str(m.get("content", ""))[:200]  # 单条截断 200 字
            parts.append(f"{role_label}: {content}")
        return "\n".join(parts)

    def get_or_generate_summary(self, current_input: str, token_budget_remaining: int) -> str:
        """Layer 2: 惰性摘要。仅在满足触发条件时生成/更新，否则返回缓存（可能为空）。

        增量式更新：只摘要 ``_summary_valid_until`` 到 recent_raw 起始之间的新消息，
        并把旧摘要一并喂给 LLM 做归并，避免每轮全量重算。
        """
        if not self._should_summarize(current_input, token_budget_remaining):
            return self._cached_summary

        recent_start = max(0, len(self.messages) - self.RECENT_TURNS * 2)
        to_summarize = self.messages[self._summary_valid_until:recent_start]
        if not to_summarize:
            return self._cached_summary

        new_content = self._format_for_summary(to_summarize)
        if self._cached_summary:
            prompt_input = f"旧摘要: {self._cached_summary}\n新对话: {new_content}"
        else:
            prompt_input = new_content

        summary = self._call_summary_llm(prompt_input)
        if summary:
            self._cached_summary = summary[: self.SUMMARY_MAX_CHARS]
            self._summary_valid_until = recent_start
        return self._cached_summary

    def _should_summarize(self, current_input: str, token_budget_remaining: int) -> bool:
        """三个条件同时满足才触发摘要生成。"""
        # 条件1: 历史超过 RECENT_TURNS 轮（否则全在 Layer 1 原文里，无需摘要）
        if len(self.messages) <= self.RECENT_TURNS * 2:
            return False
        # 条件2: 当前输入含指代标记（自包含问题无需回看历史）
        if not self._has_reference_markers(current_input):
            return False
        # 条件3: token 预算有余量（摘要最多占 800 字 ≈ 1200 token）
        if token_budget_remaining < 1500:
            return False
        return True

    _REFERENCE_MARKERS = (
        "那个", "上面", "刚才", "继续", "前面", "之前", "同样",
        "这个", "它", "他们", "那条", "该", "上述", "你說的", "你说的",
    )

    def _has_reference_markers(self, text: str) -> bool:
        return any(m in text for m in self._REFERENCE_MARKERS)

    def _call_summary_llm(self, content: str) -> str:
        """轻量 LLM 调用生成摘要。任何失败都返回空串（不阻断主流程）。"""
        try:
            from src.llm import chat_completion
            result = chat_completion(
                "用一句话概括以下法律咨询对话的要点，保留关键法条编号和术语：",
                content[:800],
            )
            return result.strip() if result else ""
        except Exception:  # noqa: BLE001 - 摘要失败不阻断主流程
            return ""

    def _format_for_summary(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            parts.append(f"{role}: {str(m.get('content', ''))[:100]}")
        return " | ".join(parts)

    def summarize_last_n(self, n: int = 3) -> str:
        """[DEPRECATED] 返回最近 N 轮的拼接摘要（纯字符串操作，不调 LLM）。

        .. deprecated:: Harness V2 Phase 3
            主流程改用分层 API（:meth:`get_recent_raw` + :meth:`get_or_generate_summary`）。
            本方法仅为向后兼容保留（旧测试 / 旧调用方），不再每轮无条件触发。

        同时收录 user/assistant 两侧消息（各截 100 字）：追问场景
        （「那第二款呢？」）既依赖用户上一问，也依赖助手上一答的落点。
        无历史时返回空串。
        """
        recent = self.messages[-n * 2:]
        prefix = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}
        parts = [
            f"{prefix.get(m['role'], m['role'])}: {str(m.get('content', ''))[:100]}"
            for m in recent
            if str(m.get("content", "")).strip()
        ]
        return " | ".join(parts)

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m["role"] == "user":
                return str(m.get("content", ""))
        return ""


# ── 模块级会话注册表（进程内 LRU，风格同 src/cache.py）────────────────
#
# 为什么需要：SessionMemory 本身只是数据对象，此前没有任何按 session_id
# 取回它的入口，harness 的多轮对话因此每轮冷启动。注册表只活在进程内、
# 有界淘汰（MAX_SESSIONS × MAX_MESSAGES_PER_SESSION），不做持久化——
# 审计/回放仍走 src/db 的 JSONL（BLUEPRINT D-12 附）。

_sessions_lock = threading.Lock()
_sessions: "OrderedDict[str, SessionMemory]" = OrderedDict()


# ── Harness V2 Phase 3: session 级并发锁（asyncio，串行化同一会话的并发请求）──
#
# 为什么用 asyncio.Lock 而非 threading.Lock：run_agent_harness 是协程，同一
# session_id 的并发请求需被串行化以避免会话记忆的读改写竞态；asyncio.Lock 在
# 事件循环内让出控制权而不阻塞线程。LRU 淘汰防止锁字典无限增长。
_session_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()
_MAX_LOCKS = 128


def get_session_lock(session_id: str) -> asyncio.Lock:
    """获取或创建 session 级别的并发锁（LRU 淘汰）。"""
    if session_id in _session_locks:
        _session_locks.move_to_end(session_id)
        return _session_locks[session_id]
    lock = asyncio.Lock()
    _session_locks[session_id] = lock
    while len(_session_locks) > _MAX_LOCKS:
        _session_locks.popitem(last=False)
    return lock


def get_or_create_session(session_id: str) -> SessionMemory:
    """按 session_id 取回会话；不存在则新建。命中时刷新 LRU 位置。"""
    with _sessions_lock:
        mem = _sessions.get(session_id)
        if mem is None:
            mem = SessionMemory(session_id=session_id)
            _sessions[session_id] = mem
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return mem


def get_session(session_id: str) -> SessionMemory | None:
    """只读取会话；不存在返回 None（不创建）。"""
    with _sessions_lock:
        mem = _sessions.get(session_id)
        if mem is not None:
            _sessions.move_to_end(session_id)
        return mem


def record_turn(
    session_id: str | None,
    *,
    user_message: str,
    assistant_message: str,
    citations: dict | None = None,
    grounding: dict | None = None,
) -> None:
    """把一轮完整对话（用户问 + 助手答）写回会话记忆。

    harness 在 finalize 时调用；session_id 为 None 时静默跳过（单轮模式）。
    """
    if not session_id:
        return
    mem = get_or_create_session(session_id)
    if user_message:
        mem.add_message("user", user_message)
    if assistant_message:
        mem.add_message(
            "assistant", assistant_message, citations=citations, grounding=grounding
        )


def session_count() -> int:
    """当前注册表里的会话数（自检/测试用）。"""
    with _sessions_lock:
        return len(_sessions)


def clear_sessions() -> None:
    """清空注册表（测试隔离用）。"""
    with _sessions_lock:
        _sessions.clear()
        _session_locks.clear()
