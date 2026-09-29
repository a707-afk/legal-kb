from __future__ import annotations
import logging
from typing import Any
from llama_index.core.schema import NodeWithScore
from src.access_control import can_access_chunk_metadata
from src.config_LEGACY_REFERENCE import Settings

logger = logging.getLogger(__name__)


def resolve_allowed_node_ids(
    settings: Settings,
    *,
    roles: list[str] | None,
    tenant_id: str | None,
    security_clearance: int,
) -> frozenset[str]:
    """基于 BM25 语料元数据（与索引节点 ID 一致）计算当前用户可检索的 node_id 集合。"""
    from src.bm25_store import _get_bm25
    try:
        _, ids, meta_lookup = _get_bm25(settings)
    except FileNotFoundError:
        logger.warning("BM25 语料缺失，无法预筛候选 ID")
        return frozenset()
    allowed: set[str] = set()
    for nid in ids:
        m = meta_lookup.get(nid) or {}
        if can_access_chunk_metadata(m, roles=roles, tenant_id=tenant_id, security_clearance=security_clearance):
            allowed.add(nid)
    logger.debug("access prefilter: allowed=%s / corpus=%s", len(allowed), len(ids))
    return frozenset(allowed)


def vector_retrieve_access_filtered(
    index: Any,
    query: str,
    top_k: int,
    settings: Settings,
    *,
    roles: list[str] | None = None,
    tenant_id: str | None = None,
    security_clearance: int = 0,
    allowed_ids: frozenset[str] | None = None,
) -> list[NodeWithScore]:
    """Query Qdrant via llama_index VectorStoreIndex with optional node_id pre-filter.

    When *allowed_ids* is provided, only results whose node_id is in the set
    are returned — this enforces tenant/role-based access control on retrieval.
    """
    if top_k < 1:
        return []
    try:
        # Request extra candidates so post-filter still yields enough results
        fetch_k = top_k * 3 if allowed_ids else top_k
        retriever = index.as_retriever(similarity_top_k=fetch_k)
        results = retriever.retrieve(query)

        # ── Access control: filter by allowed node IDs ──
        if allowed_ids is not None:
            results = [r for r in results if r.node.node_id in allowed_ids]
            logger.debug(
                "access_prefilter: %d/%d results passed access check",
                len(results), fetch_k,
            )

        return results[:top_k]
    except Exception as e:
        logger.warning("Qdrant vector retrieve failed: %s", e)
        return []