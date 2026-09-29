"""Permission Gate for Agent tools.

Enforces:
- Risk-level-based access control
- Tenant isolation (no cross-tenant access)
- Scope requirements for high-risk tools
- Prompt injection prevention
- Audit logging for all denied/approval actions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.agent.tool_registry import ToolDef, RiskLevel

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────

@dataclass
class PermissionResult:
    allowed: bool
    reason: str = ""
    requires_approval: bool = False
    risk_level: str = "low"


# ── Risk-to-action mapping ─────────────────────────────────────────

_RISK_ACTION_MAP = {
    RiskLevel.LOW: "allow",            # read_only 检索
    RiskLevel.MEDIUM: "allow_audit",   # write_internal — 自动但记审计
    RiskLevel.HIGH: "need_scope",      # 出具法律意见/代理诉讼 — 需 scope 审批
    RiskLevel.CRITICAL: "deny",        # 删除/导出/注销 — 默认拒绝或 HITL
}

# Required scopes per risk level
_REQUIRED_SCOPES = {
    "need_scope": ["legal:review"],
    "deny": ["admin:all"],
}

# High-risk parameter patterns (prompt injection detection)
_SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "sudo ",
    "you are now",
    "act as",
    "pretend",
]


def _detect_injection(params: dict[str, Any]) -> str | None:
    """Detect prompt injection in tool parameters."""
    params_str = str(params).lower()
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern in params_str:
            return f"Suspicious pattern detected: '{pattern}'"
    return None


def _check_tenant_isolation(params: dict[str, Any], allowed_tenant: str) -> str | None:
    """Check if tool parameters reference a tenant other than the user's."""
    param_tenant = params.get("tenant_id", "")
    if param_tenant and param_tenant != allowed_tenant:
        return f"Cross-tenant access denied: requested {param_tenant}, allowed {allowed_tenant}"
    return None


def check_permission(
    tool: ToolDef,
    user_context: dict[str, Any],
    params: dict[str, Any],
    tenant_id: str = "default",
) -> PermissionResult:
    """Check if a tool execution is permitted.

    Args:
        tool: The tool definition
        user_context: User's context (roles, scopes, tenant)
        params: Tool parameters
        tenant_id: Current tenant scope

    Returns:
        PermissionResult with allowed/denied and reason
    """
    # Step 1: Prompt injection check
    injection = _detect_injection(params)
    if injection:
        _audit_denied(tool.name, "injection", injection, tenant_id)
        return PermissionResult(False, reason=injection, risk_level=tool.risk_level.value)

    # Step 2: Tenant isolation check
    tenant_violation = _check_tenant_isolation(params, tenant_id)
    if tenant_violation:
        _audit_denied(tool.name, "tenant_isolation", tenant_violation, tenant_id)
        return PermissionResult(False, reason=tenant_violation, risk_level=tool.risk_level.value)

    # Step 3: Risk-level based access
    risk = tool.risk_level

    if risk == RiskLevel.LOW:
        # Read-only — always allowed
        return PermissionResult(True, risk_level="low")

    elif risk == RiskLevel.MEDIUM:
        # Write internal — allowed but audit
        _audit_allowed(tool.name, "write_internal", params, tenant_id)
        return PermissionResult(True, risk_level="medium")

    elif risk == RiskLevel.HIGH:
        # Need scope check
        user_scopes = set(user_context.get("scopes", []))
        user_roles = set(user_context.get("roles", []))
        required = set(tool.required_scopes)

        has_scope = bool(required & user_scopes)
        has_role = any("supervisor" in r.lower() or "admin" in r.lower() for r in user_roles)

        if has_scope or has_role:
            _audit_allowed(tool.name, "high_risk_scoped", params, tenant_id)
            return PermissionResult(True, risk_level="high")
        else:
            # Needs HITL approval
            _audit_needs_approval(tool.name, params, tenant_id)
            return PermissionResult(False, reason=f"Tool {tool.name} requires scope {required} or supervisor role",
                                    requires_approval=True, risk_level="high")

    elif risk == RiskLevel.CRITICAL:
        # Default deny
        user_roles = set(user_context.get("roles", []))
        if "admin" in user_roles:
            _audit_allowed(tool.name, "critical_admin_bypass", params, tenant_id)
            return PermissionResult(True, risk_level="critical")
        _audit_denied(tool.name, "critical", f"Tool {tool.name} requires admin role", tenant_id)
        return PermissionResult(False, reason=f"Tool {tool.name} requires admin role", risk_level="critical")

    return PermissionResult(False, reason="Unknown risk level")


# ── Audit logging ──────────────────────────────────────────────────

def _audit_allowed(tool_name: str, reason: str, params: dict, tenant_id: str) -> None:
    logger.info("PERMISSION_ALLOWED tool=%s reason=%s tenant=%s params=%s",
                tool_name, reason, tenant_id, str(params)[:200])
    # Fire-and-forget DB audit log
    _write_db_audit(tenant_id, "allow", f"tool:{tool_name}", detail={"reason": reason, "params_preview": str(params)[:200]})


def _audit_denied(tool_name: str, reason: str, detail: str, tenant_id: str) -> None:
    logger.warning("PERMISSION_DENIED tool=%s reason=%s detail=%s tenant=%s",
                   tool_name, reason, detail, tenant_id)
    _write_db_audit(tenant_id, "deny", f"tool:{tool_name}", detail={"reason": reason, "detail": detail}, risk_level="high")


def _audit_needs_approval(tool_name: str, params: dict, tenant_id: str) -> None:
    logger.warning("PERMISSION_NEEDS_APPROVAL tool=%s tenant=%s params=%s",
                   tool_name, tenant_id, str(params)[:200])
    _write_db_audit(tenant_id, "hitl_request", f"tool:{tool_name}", detail={"params_preview": str(params)[:200]})


def _write_db_audit(tenant_id: str, event_type: str, action: str, detail: dict | None = None, risk_level: str | None = None) -> None:
    """Schedule an async audit log write. Safe in both sync and async contexts."""
    try:
        import asyncio
        from src.audit_service import write_audit_log
        coro = write_audit_log(
            tenant_id=tenant_id,
            event_type=event_type,
            action=action,
            detail=detail,
            risk_level=risk_level,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # No running loop (e.g. sync test context) — skip silently
            coro.close()
    except Exception:
        pass  # Non-critical: don't disrupt business flow
