"""Tool Registry: centralized registration, validation, and execution dispatch.

Every tool must be registered with:
- name, description, JSON schema
- side_effect level (read_only | write_internal | external_side_effect)
- required scopes
- timeout, retry policy
- idempotency key builder
- concurrency rule

Execution MUST go through:
    ToolCall → Schema validate → Permission gate → Idempotency check → Execute → Result

Never call a tool handler directly — always go through ToolRegistry.execute().
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────

class SideEffect(Enum):
    READ_ONLY = "read_only"
    WRITE_INTERNAL = "write_internal"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class RiskLevel(Enum):
    LOW = "low"        # read_only 检索类：自动执行
    MEDIUM = "medium"  # write_internal：自动执行 + 审计
    HIGH = "high"      # 出具法律意见/代理诉讼等执业行为：需 scope/HITL
    CRITICAL = "critical"  # 删除/导出/注销等破坏性操作：默认拒绝或 HITL


@dataclass
class ToolDef:
    """Complete tool definition."""
    name: str
    description: str
    schema: dict[str, Any]
    side_effect: SideEffect = SideEffect.READ_ONLY
    risk_level: RiskLevel = RiskLevel.LOW
    required_scopes: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    max_retries: int = 1
    idempotency_window_seconds: int = 300

    # Implementation
    handler: Callable | None = None
    mock_handler: Callable | None = None

    # Eval fixtures
    eval_fixtures: list[dict] = field(default_factory=list)

    def build_idempotency_key(self, params: dict[str, Any]) -> str:
        """Build an idempotency key from tool name + sorted params."""
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
        raw = f"{self.name}:{canonical}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ToolCallResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    idempotency_key: str | None = None
    permission_denied: bool = False
    permission_reason: str | None = None


# ── Registry ───────────────────────────────────────────────────────

class ToolRegistry:
    """Central registry for all agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        # 进程内幂等缓存：重复调用不重复产生副作用（不做 Redis/DB，BLUEPRINT 第五章）
        self._idempotency_cache: dict[str, ToolCallResult] = {}

    def register(self, tool: ToolDef) -> None:
        """Register a tool definition."""
        if tool.name in self._tools:
            logger.warning("Overwriting registered tool: %s", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, risk_filter: RiskLevel | None = None) -> list[ToolDef]:
        """List all tools, optionally filtered by risk level."""
        tools = list(self._tools.values())
        if risk_filter:
            tools = [t for t in tools if t.risk_level == risk_filter]
        return tools

    def validate_params(self, tool_name: str, params: dict[str, Any]) -> list[str]:
        """Validate parameters against the tool's JSON schema. Returns list of errors."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return [f"Unknown tool: {tool_name}"]

        schema = tool.schema
        required = schema.get("function", {}).get("parameters", {}).get("required", [])
        properties = schema.get("function", {}).get("parameters", {}).get("properties", {})

        errors = []
        # Check required fields
        for req_field in required:
            if req_field not in params:
                errors.append(f"Missing required parameter: {req_field}")

        # Basic type checks
        for key, value in params.items():
            if key in properties:
                prop = properties[key]
                expected_type = prop.get("type", "string")
                if expected_type == "integer" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter {key} should be integer, got {type(value).__name__}")
                elif expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter {key} should be string, got {type(value).__name__}")

        return errors

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        user_context: dict[str, Any] | None = None,
        tenant_id: str = "default",
        skip_permission: bool = False,
    ) -> ToolCallResult:
        """Execute a tool through the full gate pipeline.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            user_context: User context for permission checking
            tenant_id: Tenant scope
            skip_permission: If True, bypass permission gate (for testing only)

        Returns:
            ToolCallResult with execution outcome
        """
        t0 = time.perf_counter()
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolCallResult(tool_name, False, error=f"Unknown tool: {tool_name}",
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        # Step 1: Schema validate
        errors = self.validate_params(tool_name, params)
        if errors:
            return ToolCallResult(tool_name, False, error="; ".join(errors),
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        # Step 2: Permission gate
        if not skip_permission:
            from src.agent.permission_gate import check_permission, PermissionResult
            perm: PermissionResult = check_permission(tool, user_context or {}, params, tenant_id)
            if not perm.allowed:
                return ToolCallResult(
                    tool_name, False,
                    error=f"Permission denied: {perm.reason}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    permission_denied=True,
                    permission_reason=perm.reason,
                )
            if perm.requires_approval:
                return ToolCallResult(
                    tool_name, False,
                    error="Requires human approval",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    permission_denied=True,
                    permission_reason="requires_approval",
                )

        # Step 3: Idempotency check（进程内缓存；不做 Redis/DB，BLUEPRINT 第五章）
        idem_key = tool.build_idempotency_key(params)
        if idem_key in self._idempotency_cache:
            cached = self._idempotency_cache[idem_key]
            logger.info("In-memory idempotency cache hit for %s/%s", tool_name, idem_key)
            return cached

        # Step 4: Execute
        handler = tool.handler
        if handler is None:
            return ToolCallResult(tool_name, False, error=f"No handler for tool: {tool_name}",
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        try:
            result = handler(params)
            # If handler is async, await it
            import asyncio
            import inspect
            if inspect.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=tool.timeout_seconds)
            # If result is a ToolResult (old format), convert
            if hasattr(result, 'success'):
                tr = ToolCallResult(
                    tool_name=tool_name,
                    success=result.success,
                    data=result.data if hasattr(result, 'data') else {},
                    error=result.error if hasattr(result, 'error') else None,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    idempotency_key=idem_key,
                )
            else:
                tr = ToolCallResult(
                    tool_name=tool_name, success=True, data=result if isinstance(result, dict) else {"result": str(result)},
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    idempotency_key=idem_key,
                )
        except asyncio.TimeoutError:
            tr = ToolCallResult(tool_name, False, error=f"Timeout after {tool.timeout_seconds}s",
                                latency_ms=tool.timeout_seconds * 1000)
        except Exception as e:
            logger.exception("Tool %s execution failed", tool_name)
            tr = ToolCallResult(tool_name, False, error=str(e),
                                latency_ms=(time.perf_counter() - t0) * 1000)

        # Step 5: 仅缓存成功结果做幂等（失败无副作用，应允许重试重新执行；见 D-13）
        if tr.success:
            self._idempotency_cache[idem_key] = tr
            if len(self._idempotency_cache) > 1000:
                oldest = list(self._idempotency_cache.keys())[:100]
                for k in oldest:
                    del self._idempotency_cache[k]

        return tr

    def clear_idempotency_cache(self) -> None:
        self._idempotency_cache.clear()


# ── Global singleton ───────────────────────────────────────────────

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_default_tools(_registry)
    return _registry


def _register_default_tools(reg: ToolRegistry) -> None:
    """Register all knowledge-base Agent tools.

      - local_search: retrieve from the local legal corpus
      - synthesize:   LLM-synthesize evidence into a cited answer
    """
    from src.agent.research_tools import (
        TOOL_LOCAL_SEARCH, TOOL_SYNTHESIZE,
        _execute_local_search, _execute_synthesize,
    )

    # All tools are read-only (no side effects on business state).
    # synthesize calls the LLM but writes nothing. Risk stays LOW so
    # the ReAct loop can call them freely without HITL.
    reg.register(ToolDef(
        name="local_search",
        description="检索本地法律法规知识库（Qdrant+BM25 混合检索）",
        schema=TOOL_LOCAL_SEARCH,
        side_effect=SideEffect.READ_ONLY, risk_level=RiskLevel.LOW,
        handler=_execute_local_search, timeout_seconds=15,
    ))
    reg.register(ToolDef(
        name="synthesize",
        description="LLM 综合多源证据为带引用的分析回答",
        schema=TOOL_SYNTHESIZE,
        side_effect=SideEffect.READ_ONLY, risk_level=RiskLevel.LOW,
        handler=_execute_synthesize, timeout_seconds=30,
    ))
