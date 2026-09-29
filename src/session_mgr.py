"""Session memory manager: multi-turn context injection for Agent conversations."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Max context window: last N messages to inject as prompt history
MAX_CONTEXT_MESSAGES = 20
# Max total chars for injected context (to avoid blowing LLM context window)
MAX_CONTEXT_CHARS = 4000


@dataclass
class SessionMemory:
    """Manages conversation history for a single chat session.

    Does NOT touch DB directly — works with Message-like dicts.
    The caller (service layer) is responsible for persistence.
    """

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)

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

    def summarize_last_n(self, n: int = 3) -> str:
        """Return a short summary of the last N exchanges for context."""
        user_msgs = [m for m in self.messages[-n*2:] if m["role"] == "user"]
        if not user_msgs:
            return ""
        return " | ".join(
            str(m["content"])[:100] for m in user_msgs[-n:]
        )

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m["role"] == "user":
                return str(m.get("content", ""))
        return ""
