"""进程内检索结果缓存（L1 精确 LRU）。

BLUEPRINT 第五章明确不做 Redis / 分布式缓存，因此这里只提供进程内 LRU：
- reindex 后调用 cache_clear()（见 vector_index.rebuild_index）；
- 热查询命中即亚秒返回，对应 PLAN D2「热检索 < 1s」验收。
不引入 TTL/Redis L2/语义 L3——那些是旧 config 里的伪需求，已随一代残留清理。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# key -> 缓存的 ScoredRetrieval 对象（进程内，不落盘）
_store: "OrderedDict[str, Any]" = OrderedDict()

# 影响检索结果、需进入缓存键的 settings 字段
_FINGERPRINT_FIELDS = (
    "embedding_model_name",
    "qdrant_collection_name",
    "qdrant_collection_name_cn",
    "rerank_enabled",
    "rerank_model",
    "rerank_candidate_top_k",
    "hybrid_bm25_enabled",
    "hybrid_fusion",
    "hybrid_rrf_k",
    "bm25_candidate_top_k",
    "query_rewrite_mode",
    "currency_rerank_enabled",
)


def _settings_fingerprint(settings: Any) -> str:
    parts = {k: getattr(settings, k, None) for k in _FINGERPRINT_FIELDS}
    return json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)


def _user_context_fingerprint(user_context: Any) -> str:
    if user_context is None:
        return "none"
    return json.dumps(
        {
            "roles": sorted(getattr(user_context, "roles", []) or []),
            "tenant_id": getattr(user_context, "tenant_id", None),
            "security_clearance": getattr(user_context, "security_clearance", None),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def build_retrieval_cache_key(
    *,
    user_query: str,
    top_k: int,
    settings: Any,
    use_query_rewrite: bool | None = None,
    user_context: Any | None = None,
    skip_domain_router: bool = False,
) -> str:
    """由查询 + top_k + 关键 settings + 权限上下文构造稳定缓存键（sha256 前 32 位）。"""
    payload = {
        "q": user_query,
        "k": top_k,
        "s": _settings_fingerprint(settings),
        "qr": use_query_rewrite,
        "uc": _user_context_fingerprint(user_context),
        "sdr": skip_domain_router,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_get_retrieval(cache_key: str, *, user_query: str | None = None, settings: Any = None) -> tuple[Any, str]:
    """命中返回 (ScoredRetrieval, "L1")；未命中或禁用返回 (None, "miss")。"""
    if settings is not None and not getattr(settings, "cache_enabled", True):
        return None, "miss"
    with _lock:
        if cache_key in _store:
            _store.move_to_end(cache_key)
            return _store[cache_key], "L1"
    return None, "miss"


def cache_put_retrieval(cache_key: str, user_query: str, result: Any, settings: Any) -> None:
    """写入缓存并按 cache_max_entries 做 LRU 淘汰。"""
    if settings is not None and not getattr(settings, "cache_enabled", True):
        return
    max_entries = int(getattr(settings, "cache_max_entries", 256) or 256)
    with _lock:
        _store[cache_key] = result
        _store.move_to_end(cache_key)
        while len(_store) > max_entries:
            _store.popitem(last=False)


def cache_clear() -> None:
    """清空进程内缓存（reindex 后必须调用，避免返回过期结果）。"""
    with _lock:
        _store.clear()


def cache_size() -> int:
    with _lock:
        return len(_store)
