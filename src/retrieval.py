"""向量 + BM25 混合召回，再 Rerank（中文法律 KB，单语）。

一代客服的电商 domain_router / retrieval_intent_boost 已删除，检索链路改为
领域无关；语言路由收敛为中文单语（见 language_router）。RouterResult 保留为
中性 stub，使下游读取 ``scored.router_result`` 的代码（cache.py / routes_rag.py）
继续可用（实际恒为 None）。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from llama_index.core.schema import NodeWithScore

from src.config_LEGACY_REFERENCE import Settings
from src.logging_utils import log_structured_event
from src.query_rewrite import resolve_retrieval_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterResult:
    """Neutral router result stub.

    The e-commerce domain router was removed. This dataclass is retained
    so ``ScoredRetrieval.router_result`` and ``cache.py`` continue to
    type-check. It is always None in practice (no domain router).
    """
    allowed_domains: tuple[str, ...] = ()
    primary_domain: str | None = None
    confidence: float = 0.0
    method: str = "disabled"
    raw_confidence: float | None = None
    domain_weights: tuple[tuple[str, float], ...] = ()
    routing_trace: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScoredRetrieval:
    """检索+重排结果。"""

    nodes: list[NodeWithScore]
    retrieval_query: str
    router_result: RouterResult | None = None
    language: str | None = None  # "zh" | "en" | "de" | "other"
    collection_used: str | None = None


def _normalize_scores_minmax(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """将单路召回分数线性缩放到 [0, 1]（一路内 min-max）。"""
    if not nodes:
        return []
    raw = [float(sn.score) if sn.score is not None else 0.0 for sn in nodes]
    lo, hi = min(raw), max(raw)
    if hi <= lo:
        normed = [1.0 if hi > 0.0 else 0.0 for _ in raw]
    else:
        span = hi - lo
        normed = [(x - lo) / span for x in raw]
    return [
        NodeWithScore(node=sn.node, score=float(n))
        for sn, n in zip(nodes, normed)
    ]


def _merge_hybrid_by_node_id(
    vector_scored: list[NodeWithScore],
    bm25_scored: list[NodeWithScore],
    *,
    normalize_scores: bool = True,
    fusion: str = "max",
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    if fusion == "rrf":
        return _merge_hybrid_by_rrf(vector_scored, bm25_scored, k=rrf_k)

    vec = (
        _normalize_scores_minmax(vector_scored)
        if normalize_scores
        else list(vector_scored)
    )
    bm25 = (
        _normalize_scores_minmax(bm25_scored)
        if normalize_scores
        else list(bm25_scored)
    )
    by_id: dict[str, NodeWithScore] = {}
    for sn in vec:
        by_id[sn.node.node_id] = sn
    for sn in bm25:
        nid = sn.node.node_id
        prev = by_id.get(nid)
        if prev is None:
            by_id[nid] = sn
            continue
        best = max(float(prev.score or 0.0), float(sn.score or 0.0))
        by_id[nid] = NodeWithScore(node=prev.node, score=best)
    merged = list(by_id.values())
    merged.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return merged


def _merge_hybrid_by_rrf(
    vector_scored: list[NodeWithScore],
    bm25_scored: list[NodeWithScore],
    *,
    k: int = 60,
) -> list[NodeWithScore]:
    """按 RRF 融合两路召回排名，避免对齐 BM25 与向量分数尺度。"""
    k = max(1, int(k))
    by_id: dict[str, tuple[Any, float]] = {}

    def add_ranked(nodes: list[NodeWithScore]) -> None:
        seen: set[str] = set()
        for rank, sn in enumerate(nodes, start=1):
            nid = sn.node.node_id
            if nid in seen:
                continue
            seen.add(nid)
            prev = by_id.get(nid)
            score = 1.0 / (k + rank)
            if prev is None:
                by_id[nid] = (sn.node, score)
            else:
                node, total = prev
                by_id[nid] = (node, total + score)

    add_ranked(vector_scored)
    add_ranked(bm25_scored)

    merged = [
        NodeWithScore(node=node, score=score)
        for node, score in by_id.values()
    ]
    merged.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return merged



def _log_retrieve_event(
    trace_id: str | None,
    *,
    hits: int,
    retrieval_query: str,
    router_result: RouterResult | None,
    degraded: str | None = None,
) -> None:
    primary_domain = None
    if router_result is not None:
        primary_domain = router_result.primary_domain
    log_structured_event(
        trace_id,
        "retrieve",
        hits=hits,
        retrieval_query=(retrieval_query or "")[:500],
        primary_domain=primary_domain,
        degraded=degraded,
    )


def retrieve_scored_nodes(
    index: Any,
    user_query: str,
    top_k: int,
    settings: Settings,
    *,
    use_query_rewrite: bool | None = None,
    user_context: Any | None = None,
    skip_domain_router: bool = False,
    trace_id: str | None = None,
    trace: "Any | None" = None,
) -> ScoredRetrieval:
    from src.cache import (
        build_retrieval_cache_key,
        cache_get_retrieval,
        cache_put_retrieval,
    )
    from src.telemetry import trace_span

    cache_key = build_retrieval_cache_key(
        user_query=user_query,
        top_k=top_k,
        settings=settings,
        use_query_rewrite=use_query_rewrite,
        user_context=user_context,
        skip_domain_router=skip_domain_router,
    )
    with trace_span("retrieve_scored_nodes", trace_id=trace_id, top_k=top_k):
        cached, level = cache_get_retrieval(
            cache_key, user_query=user_query, settings=settings
        )
        if cached is not None:
            logger.debug("检索缓存命中 (%s): %s", level, (user_query or "")[:50])
            return cached

        result = _retrieve_scored_nodes_impl(
            index,
            user_query,
            top_k,
            settings,
            use_query_rewrite=use_query_rewrite,
            user_context=user_context,
            skip_domain_router=skip_domain_router,
            trace_id=trace_id,
            trace=trace,
        )
        cache_put_retrieval(cache_key, user_query, result, settings)
        return result


def _retrieve_scored_nodes_impl(
    index: Any,
    user_query: str,
    top_k: int,
    settings: Settings,
    *,
    use_query_rewrite: bool | None = None,
    user_context: Any | None = None,
    skip_domain_router: bool = False,
    trace_id: str | None = None,
    trace: "Any | None" = None,
) -> ScoredRetrieval:
    def _tr(stage: str, **fields):
        if trace is not None:
            trace.record(stage, **fields)

    # ── 语言检测 + 双索引路由 ──
    from src.language_router import detect_language, get_collection_for_lang

    rq = resolve_retrieval_query(
        user_query,
        settings,
        use_rewrite=use_query_rewrite,
        trace_id=trace_id,
    )

    lang = detect_language(rq)
    lang_route = get_collection_for_lang(lang, settings)
    logger.info("语言路由: %s → collection=%s", lang, lang_route.collection_name)

    candidate_k = (
        max(top_k, settings.rerank_candidate_top_k)
        if settings.rerank_enabled
        else top_k
    )
    _tr("route", language=lang, collection=lang_route.collection_name,
        retrieval_query=rq[:200], candidate_k=candidate_k,
        rerank_enabled=bool(settings.rerank_enabled),
        fusion=getattr(settings, "hybrid_fusion", "max"))

    # 中文单语单集合：直接用传入的索引（build_index 建的就是它）。
    # 一代的「中英双集合路由」已收敛——不再有 get_vector_index_cn 回退与每查询告警。
    idx = index

    rr: RouterResult | None = None
    # domain_router was removed during the legal-KB refactoring; rr stays None.

    allowed_ids: frozenset[str] | None = None
    if user_context is not None:
        from src.access_prefilter import resolve_allowed_node_ids

        allowed_ids = resolve_allowed_node_ids(
            settings,
            roles=user_context.roles,
            tenant_id=user_context.tenant_id,
            security_clearance=user_context.security_clearance,
        )
        if not allowed_ids:
            _log_retrieve_event(trace_id, hits=0, retrieval_query=rq, router_result=rr)
            return ScoredRetrieval(
                nodes=[], retrieval_query=rq, router_result=rr,
                language=lang, collection_used=lang_route.collection_name,
            )

    # Parallel vector + BM25 search（**带超时预算**，见 D-14 附三）
    from concurrent.futures import ThreadPoolExecutor, wait as _wait

    vector_scored: list[NodeWithScore] = []
    bm25_nodes: list[NodeWithScore] = []
    _lock = threading.Lock()

    def _do_vector_search():
        nonlocal vector_scored
        if allowed_ids is not None:
            from src.access_prefilter import vector_retrieve_access_filtered
            vs = vector_retrieve_access_filtered(
                idx, rq, candidate_k, settings,
                roles=user_context.roles, tenant_id=user_context.tenant_id,
                security_clearance=user_context.security_clearance, allowed_ids=allowed_ids,
            )
        else:
            retriever = idx.as_retriever(similarity_top_k=candidate_k)
            vs = retriever.retrieve(rq)
        with _lock:
            vector_scored = list(vs)

    def _do_bm25_search():
        if not settings.hybrid_bm25_enabled:
            return
        try:
            from pathlib import Path

            from src.bm25_store import bm25_search, node_with_score_from_bm25, _get_bm25
            bm25_path = getattr(settings, "bm25_corpus_path_cn", "data/bm25_cn_corpus.jsonl") if lang == "zh" else settings.bm25_corpus_path
            if not Path(bm25_path).is_file():
                # 中文专属语料缺失时回退主语料（本仓库 legal 语料即中文文书），
                # 避免 zh 查询的 BM25 分支整体静默失效（见 docs/FIX-retrieval-hang-20260927.md）。
                fallback = settings.bm25_corpus_path
                if Path(fallback).is_file():
                    logger.info("BM25 语料 %s 缺失，回退主语料 %s", bm25_path, fallback)
                    bm25_path = fallback
            hits = bm25_search(settings, rq, settings.bm25_candidate_top_k, allowed_ids=allowed_ids, corpus_path=bm25_path)
            _, _, meta = _get_bm25(settings, corpus_path=bm25_path)
            bn: list[NodeWithScore] = []
            for nid, bsc in hits:
                nws = node_with_score_from_bm25(nid, bsc, meta)
                if nws is not None:
                    bn.append(nws)
            with _lock:
                bm25_nodes.extend(bn)
        except FileNotFoundError:
            logger.warning("BM25 语料未找到，仅使用向量召回")
        except Exception:
            logger.exception("BM25 分支失败，回退为纯向量候选")

    # ⚠️ 不能写成 `with ThreadPoolExecutor(...)`：`__exit__` 会 join() 所有线程，
    # 即使给了 wait(timeout=)，退出 with 块时**仍会等慢的那一路**——那正是
    # 一代"首次检索挂 150s+"的成因。所以手动管理，超时后用 shutdown(wait=False) 放行。
    parallel_timeout = float(getattr(settings, "retrieval_parallel_timeout_seconds", 20.0) or 0.0)
    degraded_paths: list[str] = []
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        f_vec = pool.submit(_do_vector_search)
        f_bm25 = pool.submit(_do_bm25_search)
        done, not_done = _wait(
            [f_vec, f_bm25],
            timeout=parallel_timeout if parallel_timeout > 0 else None,
        )
        for f, name in ((f_vec, "vector"), (f_bm25, "bm25")):
            if f in done:
                exc = f.exception()
                if exc is not None:
                    degraded_paths.append(f"{name}:error")
                    logger.warning("并行检索子任务 %s 失败: %s", name, exc)
            else:
                degraded_paths.append(f"{name}:timeout")
                logger.warning("并行检索子任务 %s 超时（>%.1fs），降级为单路结果", name, parallel_timeout)
    finally:
        pool.shutdown(wait=False)  # 不等慢的那一路；运行中的线程会在后台自然结束

    # 在锁内取快照，避免后台线程继续写入造成读到半成品
    with _lock:
        vector_snapshot = list(vector_scored)
        bm25_snapshot = list(bm25_nodes)
    vector_scored, bm25_nodes = vector_snapshot, bm25_snapshot
    if trace is not None:
        trace.stage_candidates("vector", vector_scored)
        trace.stage_candidates("bm25", bm25_nodes)
        trace.record("parallel", vector=len(vector_scored), bm25=len(bm25_nodes),
                     timeout_s=parallel_timeout, degraded=",".join(degraded_paths) or None)
    if degraded_paths:
        _log_retrieve_event(
            trace_id, hits=0, retrieval_query=rq, router_result=rr,
            degraded=",".join(degraded_paths),
        )

    if bm25_nodes:
        merged = _merge_hybrid_by_node_id(
            vector_scored, bm25_nodes,
            normalize_scores=getattr(settings, "hybrid_score_normalize", True),
            fusion=getattr(settings, "hybrid_fusion", "max"),
            rrf_k=getattr(settings, "hybrid_rrf_k", 60),
        )
        logger.debug("hybrid: vec=%s bm25=%s merged=%s fusion=%s lang=%s",
            len(vector_scored), len(bm25_nodes), len(merged),
            getattr(settings, "hybrid_fusion", "max"), lang)
    else:
        merged = list(vector_scored)
    if trace is not None:
        trace.stage_candidates("fuse", merged, fusion=getattr(settings, "hybrid_fusion", "max"),
                               vector=len(vector_scored), bm25=len(bm25_nodes))

    if not merged:
        _log_retrieve_event(trace_id, hits=0, retrieval_query=rq, router_result=rr)
        return ScoredRetrieval(
            nodes=[], retrieval_query=rq, router_result=rr,
            language=lang, collection_used=lang_route.collection_name,
        )

    if (
        user_context is not None
        and getattr(settings, "access_post_filter_safety_net", False)
    ):
        from src.access_control import filter_nodes_by_access

        merged = filter_nodes_by_access(
            merged,
            roles=user_context.roles,
            tenant_id=user_context.tenant_id,
            security_clearance=user_context.security_clearance,
        )
        if not merged:
            _log_retrieve_event(trace_id, hits=0, retrieval_query=rq, router_result=rr)
            return ScoredRetrieval(
                nodes=[], retrieval_query=rq, router_result=rr,
                language=lang, collection_used=lang_route.collection_name,
            )

    # retrieval_intent_boost was removed during the legal-KB refactoring.
    # Rerank operates directly on the merged hybrid candidates.
    # (The domain hard-filter block was removed with domain_router.)

    if not merged:
        _log_retrieve_event(trace_id, hits=0, retrieval_query=rq, router_result=rr)
        return ScoredRetrieval(
            nodes=[], retrieval_query=rq, router_result=rr,
            language=lang, collection_used=lang_route.collection_name,
        )

    # 早退门控（生产：语料外/明显不相关查询快速拒答）——rerank 是最贵的一步，
    # 若向量 top-1 余弦已低到“必被后置 similarity gate 拒”，就跳过 rerank 直接返回。
    best_vector_score = max((float(sn.score or 0.0) for sn in vector_scored), default=0.0)
    early_gate = (
        settings.rerank_enabled
        and getattr(settings, "retrieval_early_gate_enabled", False)
        and best_vector_score < float(getattr(settings, "retrieval_early_gate_threshold", 0.30))
    )
    if trace is not None:
        trace.record("vector_top1", best_vector_score=round(best_vector_score, 4), early_gate=early_gate)

    if early_gate:
        nodes = merged[:top_k]
    elif settings.rerank_enabled:
        from src.rerank import rerank_nodes

        nodes = rerank_nodes(rq, merged, top_n=top_k, settings=settings)
        if trace is not None:
            trace.stage_candidates("rerank", nodes, reranked_from=len(merged))
    else:
        nodes = merged[:top_k]

    # 时效重排（D-06，独立可开关）：现行有效加权 / 未生效·废止降权 / 时间旅行按时间过滤
    from src.currency import apply_currency_rerank
    nodes = apply_currency_rerank(nodes, rq, settings, trace=trace)

    out = ScoredRetrieval(
        nodes=nodes, retrieval_query=rq, router_result=rr,
        language=lang, collection_used=lang_route.collection_name,
    )
    _log_retrieve_event(
        trace_id, hits=len(out.nodes), retrieval_query=rq, router_result=rr
    )
    return out
