from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.citation import citation_overlap_ratio, sentence_level_grounding
from src.config_LEGACY_REFERENCE import get_settings
from src.retrieval import RouterResult, retrieve_scored_nodes
from src.input_sanitizer import InputGuard
from src.vector_index import get_vector_index
from src.llm import chat_completion
from src.logging_utils import log_structured_event
from src.gates import evaluate_similarity_gate
from src.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkHit,
    CitationBlock,
    RetrieveRequest,
    RetrieveResponse,
)
from src.sse import chunk_text, format_sse_event

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


def _heading_from_meta(meta: dict[str, Any]) -> str | None:
    return meta.get("header_path") or meta.get("heading") or None


def _retrieval_query_field(user_query: str, effective: str) -> str | None:
    if (effective or "").strip() == (user_query or "").strip():
        return None
    return effective


def _router_trace_dict(rr: RouterResult | None) -> dict[str, Any] | None:
    if rr is None:
        return None
    base: dict[str, Any] = {
        "allowed_domains": list(rr.allowed_domains),
        "primary_domain": rr.primary_domain,
        "confidence": rr.confidence,
        "method": rr.method,
        "raw_confidence": rr.raw_confidence,
        "domain_weights": {d: float(w) for d, w in rr.domain_weights},
    }
    if rr.routing_trace is not None:
        base["routing_trace"] = rr.routing_trace
    return base


def _chunks_from_scored(scored) -> list[ChunkHit]:
    chunks: list[ChunkHit] = []
    for sn in scored:
        node = sn.node
        meta = dict(node.metadata or {})
        chunks.append(
            ChunkHit(
                text=node.get_content(),
                score=float(sn.score) if sn.score is not None else None,
                file_path=meta.get("file_path"),
                file_name=meta.get("file_name"),
                heading=_heading_from_meta(meta),
                node_id=node.node_id,
                domain=meta.get("domain"),
            )
        )
    return chunks


def _log_event(
    trace_id: str | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    log_structured_event(trace_id, event, **payload)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
    tid = getattr(request.state, "trace_id", None)
    settings = get_settings()

    # Endpoint-level InputGuard check (defense-in-depth on top of HTTP middleware)
    if getattr(settings, "input_guard_endpoint_enabled", True):
        guard = InputGuard.check(req.query)
        if guard.blocked:
            _log_event(tid, "retrieve", {"query": req.query[:500], "input_guard_blocked": True, "threats": guard.threats})
            return RetrieveResponse(
                query=req.query,
                retrieval_query=None,
                chunks=[],
                gate_passed=False,
                error_code="INPUT_GUARD",
                behavior="refuse",
                refusal_reason_code="INPUT_GUARD",
                ranked_quality_scores=[],
                router_trace=None,
                trace_id=tid,
            )
        if guard.threats:
            logger.warning("retrieve: InputGuard threats (non-blocking) tid=%s threats=%s", tid, guard.threats)

    try:
        index = get_vector_index()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    sr = retrieve_scored_nodes(
        index,
        req.query,
        req.top_k,
        settings,
        use_query_rewrite=req.use_query_rewrite,
        user_context=req.user_context,
        skip_domain_router=req.skip_domain_router,
        trace_id=tid,
    )
    scored = sr.nodes
    chunks = _chunks_from_scored(scored)
    rq_field = _retrieval_query_field(req.query, sr.retrieval_query)
    rt_dict = _router_trace_dict(sr.router_result)

    if not scored:
        return RetrieveResponse(
            query=req.query,
            retrieval_query=rq_field,
            chunks=chunks,
            gate_passed=False,
            error_code="NO_RESULTS",
            ranked_quality_scores=[],
            router_trace=rt_dict,
            trace_id=tid,
        )

    gate = evaluate_similarity_gate(scored, settings, trace_id=tid)
    return RetrieveResponse(
        query=req.query,
        retrieval_query=rq_field,
        chunks=chunks,
        gate_passed=gate.passed,
        error_code=None if gate.passed else gate.error_code,
        ranked_quality_scores=gate.ranked_scores,
        router_trace=rt_dict,
        trace_id=tid,
    )


SYS_PROMPT = r"""你是技术研究分析助手，必须严格执行以下规则：

## 核心规则（违反将导致处罚）
1. 仅使用参考资料中的信息回答问题，禁止编造任何事实、性能数据或基准。
2. 如果参考资料中没有足够的信息回答用户问题，必须明确回答「根据当前资料无法回答这个问题」。不要猜测或类推。
3. 回答必须为每个事实陈述标注引用 [1]、[2] 等，对应参考资料片段序号。
4. 禁止给出参考资料之外的具体数值、基准结果、版本号等细节。不确定的一律说无法确认。
5. 回答要客观、准确、有条理，使用中文，适合技术人员阅读。

## 正确示例
- 正确：
  根据参考资料，Qdrant 在 1M 向量规模下 HNSW 检索延迟约 5ms[1]。它支持 payload filter 在向量检索时预过滤[2]。如需更详细的对比，建议补充检索更多资料。
- 错误：
  Qdrant 比 Milvus 快 3 倍（资料中无此对比数据）。
- 正确：
  根据当前资料无法回答这个问题（资料中未提及）。
"""  # noqa: E501


def _execute_chat(req: ChatRequest, tid: str | None) -> ChatResponse:
    """同步执行 /chat 核心逻辑，供 JSON 与 SSE 共用。"""
    settings = get_settings()

    # Endpoint-level InputGuard check (defense-in-depth on top of HTTP middleware)
    if getattr(settings, "input_guard_endpoint_enabled", True):
        guard = InputGuard.check(req.query)
        if guard.blocked:
            _log_event(tid, "chat", {"query": req.query[:500], "input_guard_blocked": True, "threats": guard.threats})
            return ChatResponse(
                query=req.query,
                retrieval_query=None,
                answer="请求包含不安全内容，已被拒绝。",
                citations=[],
                chunks_used=0,
                refused=True,
                error_code="INPUT_GUARD",
                behavior="refuse",
                refusal_reason_code="INPUT_GUARD",
                ranked_quality_scores=[],
                router_trace=None,
                trace_id=tid,
                citation_overlap_ratio=None,
            )
        if guard.threats:
            logger.warning("chat: InputGuard threats (non-blocking) tid=%s threats=%s", tid, guard.threats)

    try:
        index = get_vector_index()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    sr = retrieve_scored_nodes(
        index,
        req.query,
        req.top_k,
        settings,
        use_query_rewrite=req.use_query_rewrite,
        user_context=req.user_context,
        skip_domain_router=req.skip_domain_router,
        trace_id=tid,
    )
    scored = sr.nodes
    rq_field = _retrieval_query_field(req.query, sr.retrieval_query)
    rt_dict = _router_trace_dict(sr.router_result)

    if not scored:
        _log_event(tid, "chat", {"query": req.query[:500], "refused": "NO_RESULTS"})
        return ChatResponse(
            query=req.query,
            retrieval_query=rq_field,
            answer=settings.refusal_no_results,
            citations=[],
            chunks_used=0,
            refused=True,
            error_code="NO_RESULTS",
            ranked_quality_scores=[],
            router_trace=rt_dict,
            trace_id=tid,
            citation_overlap_ratio=None,
        )

    gate = evaluate_similarity_gate(scored, settings, trace_id=tid)
    if not gate.passed:
        _log_event(
            tid,
            "chat",
            {"query": req.query[:500], "refused": gate.error_code},
        )
        return ChatResponse(
            query=req.query,
            retrieval_query=rq_field,
            answer=settings.refusal_gate_fail,
            citations=[],
            chunks_used=0,
            refused=True,
            error_code=gate.error_code or "GATE_FAIL",
            ranked_quality_scores=gate.ranked_scores,
            router_trace=rt_dict,
            trace_id=tid,
            citation_overlap_ratio=None,
        )

    parts: list[str] = []
    cites: list[CitationBlock] = []
    chunk_plain: list[str] = []
    chunks_with_meta: list[dict] = []
    for i, sn in enumerate(scored):
        node = sn.node
        meta = dict(node.metadata or {})
        text = node.get_content()
        chunk_plain.append(text)
        chunks_with_meta.append(
            {
                "text": text,
                "node_id": node.node_id,
                "file_name": meta.get("file_name"),
                "file_path": meta.get("file_path"),
            }
        )
        fn = meta.get("file_name") or ""
        fp = meta.get("file_path") or ""
        hd = _heading_from_meta(meta)
        parts.append(
            f"[{i + 1}] 文件: {fn} ({fp})\n标题路径: {hd or '无'}\n{text}\n"
        )
        cites.append(
            CitationBlock(
                index=i + 1,
                file_path=fp or None,
                file_name=fn or None,
                heading=hd,
                excerpt=text[:400] + ("…" if len(text) > 400 else ""),
            )
        )

    user_block = (
        "参考资料：\n\n"
        + "\n---\n".join(parts)
        + "\n\n用户问题：\n"
        + req.query.strip()
    )
    try:
        answer = chat_completion(SYS_PROMPT, user_block)
    except RuntimeError as e:
        logger.exception("LLM 调用失败")
        raise HTTPException(status_code=500, detail=str(e)) from e

    ov = citation_overlap_ratio(answer, chunk_plain)
    grounding_report = sentence_level_grounding(answer, chunks_with_meta)
    grounding_dict = grounding_report.to_dict()
    _log_event(
        tid,
        "chat",
        {
            "query": req.query[:500],
            "chunks_used": len(scored),
            "citation_overlap_ratio": ov,
            "grounding_passed": grounding_dict.get("passed"),
            "unsupported_sentence_rate": grounding_dict.get("unsupported_sentence_rate"),
            "router": rt_dict,
        },
    )
    return ChatResponse(
        query=req.query,
        retrieval_query=rq_field,
        answer=answer,
        citations=cites,
        chunks_used=len(scored),
        refused=False,
        error_code=None,
        ranked_quality_scores=gate.ranked_scores,
        router_trace=rt_dict,
        trace_id=tid,
        citation_overlap_ratio=ov,
        grounding=grounding_dict,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    tid = getattr(request.state, "trace_id", None)
    return _execute_chat(req, tid)


def _chat_stream_events(req: ChatRequest, tid: str | None) -> Iterator[str]:
    # Yield progress: retrieving
    yield format_sse_event("progress", {"stage": "retrieve", "message": "检索知识库…"})
    try:
        resp = _execute_chat(req, tid)
    except HTTPException as e:
        yield format_sse_event("error", {"message": str(e.detail), "status_code": e.status_code})
        return
    except Exception as e:
        logger.exception("chat/stream 失败")
        yield format_sse_event("error", {"message": str(e)})
        return

    # Yield progress: generating
    yield format_sse_event("progress", {
        "stage": "generate",
        "message": f"已找到 {resp.chunks_used} 个相关片段，正在生成回复…",
        "chunks_used": resp.chunks_used,
    })

    answer = resp.answer or ""
    if answer and not resp.refused:
        for piece in chunk_text(answer):
            yield format_sse_event("token", {"text": piece})

    yield format_sse_event("done", resp.model_dump(mode="json"))


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """SSE 流式问答：token 增量 + done（完整 ChatResponse 摘要）。"""
    tid = getattr(request.state, "trace_id", None)

    async def _gen():
        for chunk in _chat_stream_events(req, tid):
            yield chunk
            await asyncio.sleep(0)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
