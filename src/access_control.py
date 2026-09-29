"""Chunk metadata 与调用方 user_ctx 的访问控制（租户、密级、audience）。"""
from __future__ import annotations

from typing import Any

# 越高越敏；用户 clearance 数值需 >= 文档密级 rank 才能阅读
_SECURITY_LEVEL_RANK: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
    "secret": 4,
}

# 企业共享知识库默认租户；调用方未传 tenant_id 时仍可检索带此标签的 chunk
_SHARED_CORP_TENANT = "corp-default"

# API 角色 → 语料 audience 标签（精确匹配外的等价标签，避免 support / support_agent 分裂）
_ROLE_TO_AUDIENCE_TAGS: dict[str, frozenset[str]] = {
    "support": frozenset(
        {
            "support",
            "support_agent",
            "team_lead",
            "support_tl",
            "employee",
            "success",
        }
    ),
    "qa": frozenset(
        {
            "qa",
            "quality_assurance",
            "security_engineer",
            "kb_admin",
        }
    ),
}


def _parse_audience_tags(audience: str) -> set[str]:
    return {
        t.strip().lower()
        for t in str(audience).replace("，", ",").split(",")
        if t.strip()
    }


def expand_roles_for_audience_match(roles: list[str] | None) -> set[str]:
    """将 UserContext.roles 展开为可与 chunk.audience 求交的标签集合。"""
    out: set[str] = set()
    for r in roles or []:
        key = str(r).strip().lower()
        if not key:
            continue
        out.add(key)
        out.update(_ROLE_TO_AUDIENCE_TAGS.get(key, ()))
    return out

def _rank_security(level: str | None) -> int:
    if not level:
        return 0
    key = str(level).strip().lower()
    return _SECURITY_LEVEL_RANK.get(key, 1)


def can_access_chunk_metadata(
    meta: dict[str, Any],
    *,
    roles: list[str] | None,
    tenant_id: str | None,
    security_clearance: int,
) -> bool:
    """
    roles / tenant_id / security_clearance 来自 UserContext（全空/默认=尽量放行仅靠密级）。
    meta：security_level, audience, tenant_id（可选）。
    """
    chunk_tenant = meta.get("tenant_id") or meta.get("tenant")
    if chunk_tenant:
        ct = str(chunk_tenant).strip()
        ut = str(tenant_id).strip() if tenant_id else ""
        if not ut:
            if ct != _SHARED_CORP_TENANT:
                return False
        elif ct != ut:
            return False

    need = _rank_security(meta.get("security_level"))
    if security_clearance < need:
        return False

    aud = meta.get("audience") or ""
    if not aud or str(aud).strip().lower() in ("all", "any", "everyone"):
        return True
    if not roles:
        return False
    tags = _parse_audience_tags(aud)
    user = expand_roles_for_audience_match(roles)
    return bool(tags & user)


def filter_nodes_by_access(
    nodes,
    *,
    roles: list[str] | None,
    tenant_id: str | None,
    security_clearance: int,
):
    """输入 NodeWithScore 列表，过滤无权访问的节点。"""
    from llama_index.core.schema import NodeWithScore

    out: list[NodeWithScore] = []
    for sn in nodes:
        meta = dict(sn.node.metadata or {})
        if can_access_chunk_metadata(
            meta,
            roles=roles,
            tenant_id=tenant_id,
            security_clearance=security_clearance,
        ):
            out.append(sn)
    return out


def filter_nodes_by_domain(
    nodes,
    allowed_domains: tuple[str, ...] | list[str],
    *,
    strict: bool = False,
):
    """若 allowed_domains 非空，仅保留 metadata.domain 命中的节点（无 domain：strict=False 时保留）。"""
    from llama_index.core.schema import NodeWithScore

    if not allowed_domains:
        return nodes
    allow = {str(d).strip().lower() for d in allowed_domains}
    out: list[NodeWithScore] = []
    for sn in nodes:
        d = (sn.node.metadata or {}).get("domain")
        if not d:
            if not strict:
                out.append(sn)
            continue
        if str(d).strip().lower() in allow:
            out.append(sn)
    return out
