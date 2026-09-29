"""Knowledge-base tools for the legal compliance KB Agent.

The ReAct loop can call:

1. ``local_search``  — retrieve from the local legal corpus (Qdrant+BM25)
2. ``synthesize``    — LLM-synthesize gathered facts into a cited answer

All tools share a ``ToolResult`` contract and are registered in
``tool_registry._register_default_tools``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool schema definitions (OpenAI function-calling compatible) ──

TOOL_LOCAL_SEARCH = {
    "type": "function",
    "function": {
        "name": "local_search",
        "description": (
            "检索本地法律法规知识库（向量+BM25 混合检索）。覆盖法律、行政法规、"
            "司法解释、案例与常见问答四个分库。返回带引用编号的相关片段。"
            "适合回答'XX 法第 X 条的内容是什么''XX 情形适用哪条法律'类问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询词，优先使用法律术语与关键词（如 '劳动合同 试用期 期限 上限'）。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "返回的片段数量",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SYNTHESIZE = {
    "type": "function",
    "function": {
        "name": "synthesize",
        "description": (
            "将已收集的多条检索结果综合成一段连贯的、带引用标注 [1][2] 的分析回答。"
            "用于在 local_search 收集足够证据后，生成最终回复。"
            "调用此工具前应已通过 local_search 收集至少 2 条相关片段。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "要回答的法律问题",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "已收集的证据片段列表（来自 local_search 的结果）",
                },
            },
            "required": ["question", "evidence"],
        },
    },
}


# ── Tool result contract ──────────────────────────────────────────

@dataclass
class ToolResult:
    """Unified result for all knowledge-base tools."""
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ── Tool implementations ──────────────────────────────────────────

def _execute_local_search(args: dict[str, Any]) -> ToolResult:
    """检索本地法律法规知识库。

    复用现有 retrieve_scored_nodes（Qdrant + BM25 hybrid + rerank + gate），
    把 NodeWithScore 结果转为带引用编号的片段列表。
    """
    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k", 5))
    if not query:
        return ToolResult("local_search", False, error="query is required")

    try:
        from src.config_LEGACY_REFERENCE import get_settings
        from src.vector_index import get_vector_index
        from src.retrieval import retrieve_scored_nodes

        settings = get_settings()
        index = get_vector_index()
        if index is None:
            return ToolResult("local_search", False, error="Vector index not loaded")

        scored = retrieve_scored_nodes(
            index=index,
            user_query=query,
            top_k=top_k,
            settings=settings,
            skip_domain_router=True,  # domain router was deleted; always skip
        )

        snippets: list[dict[str, Any]] = []
        for i, sn in enumerate(scored.nodes[:top_k], start=1):
            node = sn.node
            meta = dict(node.metadata or {})
            text = (node.get_content() or "").strip()
            snippets.append({
                "index": i,
                "citation_id": node.node_id or f"src-{i}",
                "file_name": meta.get("file_name", ""),
                "title": meta.get("title", ""),
                # 键名是 heading_path（chunking 写入），不是 header_path；
                # 取错键会让标题恒为空，生成侧无法定位"第几条"。
                "heading": meta.get("heading_path") or meta.get("header_path") or meta.get("heading") or "",
                # 时效元数据必须随片段一起交给生成侧：Agent 要判"现行有效/尚未生效/未标注"，
                # 只能靠这三个字段；不带就等于让 LLM 凭空猜时效（实测 E/F 类全挂）。
                "status_code": meta.get("status_code"),
                "status_label": meta.get("status_label", ""),
                "effective_date": meta.get("effective_date", ""),
                "score": round(float(sn.score or 0.0), 4),
                "text": text[:800],  # cap snippet length for context window
            })

        return ToolResult(
            "local_search",
            success=True,
            data={
                "query": query,
                "retrieval_query": scored.retrieval_query,
                "count": len(snippets),
                "snippets": snippets,
            },
        )
    except Exception as e:
        logger.exception("local_search failed for query='%s'", query[:80])
        return ToolResult("local_search", False, error=str(e))


def _execute_synthesize(args: dict[str, Any]) -> ToolResult:
    """LLM 综合多条证据为带引用的回答。

    输入：question + evidence（来自 local_search 的 snippets text）。
    输出：一段带 [1][2] 引用标注的分析回答。
    """
    question = str(args.get("question", "")).strip()
    evidence = args.get("evidence") or []
    if not question:
        return ToolResult("synthesize", False, error="question is required")
    if not evidence or not isinstance(evidence, list):
        return ToolResult("synthesize", False, error="evidence (non-empty list) is required")

    try:
        from src.llm import chat_completion

        # Build numbered evidence block
        evidence_block = "\n\n".join(
            f"[{i+1}] {str(ev).strip()}" for i, ev in enumerate(evidence)
        )

        system = (
            "你是一个法律合规分析综合器。根据给定的法律问题和检索到的证据片段，"
            "生成一段连贯、客观、带引用标注的分析回答。\n"
            "规则：\n"
            "1. 每个事实陈述必须标注引用 [1][2] 等，对应证据序号\n"
            "2. 只使用证据中的信息，禁止编造法条内容\n"
            "3. 证据不足时明确说明，不得推测\n"
            "4. 每条证据抬头带时效标注（现行有效/已修订/尚未生效/已废止/未标注 + 生效日期）："
            "标注「尚未生效」必须说明尚未生效并给出生效日期；标注「未标注」必须说明"
            "「时效状态未标注、无法确定是否现行有效」，不得断言现行有效\n"
            "5. 输出仅供参考，不构成法律意见"
        )
        user = (
            f"法律问题：{question}\n\n"
            f"证据片段：\n{evidence_block}\n\n"
            "请生成带引用的分析回答："
        )

        answer = chat_completion(system, user)
        if not answer or len(answer) < 10:
            return ToolResult("synthesize", False, error="LLM returned empty answer")

        return ToolResult(
            "synthesize",
            success=True,
            data={
                "question": question,
                "answer": answer,
                "evidence_count": len(evidence),
            },
        )
    except Exception as e:
        logger.exception("synthesize failed for question='%s'", question[:80])
        return ToolResult("synthesize", False, error=str(e))


# ── Dispatch table ────────────────────────────────────────────────

RESEARCH_TOOL_DISPATCH: dict[str, Any] = {
    "local_search": _execute_local_search,
    "synthesize": _execute_synthesize,
}


def execute_research_tool(tool_name: str, args: dict[str, Any]) -> ToolResult:
    """Dispatch a knowledge-base tool call by name."""
    handler = RESEARCH_TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return ToolResult(tool_name, False, error=f"Unknown tool: {tool_name}")
    try:
        return handler(args)
    except Exception as e:
        logger.exception("Tool %s failed", tool_name)
        return ToolResult(tool_name, False, error=str(e))
