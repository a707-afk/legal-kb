"""检索重排：支持 Qwen3-Reranker（CausalLM）与 sentence-transformers CrossEncoder。"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

from src.config_LEGACY_REFERENCE import Settings

logger = logging.getLogger(__name__)


def infer_rerank_backend(model_path_or_id: str) -> str:
    p = Path(model_path_or_id)
    if p.is_dir() and (p / "config.json").is_file():
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "cross_encoder"
        if any("CausalLM" in str(a) for a in (cfg.get("architectures") or [])):
            return "qwen3_causal"
    return "cross_encoder"


@lru_cache(maxsize=2)
def _get_cross_encoder(model_path: str) -> CrossEncoder:
    """Load CrossEncoder from local path with caching."""
    kw = {"local_files_only": True}
    path = Path(model_path)
    if path.is_dir():
        kw["trust_remote_code"] = True
    return CrossEncoder(model_path, **kw)


def rerank_nodes(
    query: str,
    scored: list[NodeWithScore],
    top_n: int,
    settings: Settings,
) -> list[NodeWithScore]:
    if not scored or top_n < 1:
        return []
    model = settings.rerank_model
    backend = settings.rerank_backend.strip().lower()
    if backend == "auto":
        backend = infer_rerank_backend(model)
        logger.debug("rerank backend auto -> %s", backend)
    if backend == "qwen3_causal":
        from src.qwen_rerank import rerank_nodes_qwen

        try:
            return rerank_nodes_qwen(
                query,
                scored,
                top_n,
                model,
                max_length=settings.qwen_rerank_max_length,
                batch_size=settings.qwen_rerank_batch_size,
                instruction=settings.qwen_rerank_instruction,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "not enough memory" in msg or "alloc_cpu" in msg:
                logger.warning("Qwen rerank 内存不足，回退为向量/BM25 分数排序: %s", e)
                merged = sorted(
                    scored, key=lambda sn: float(sn.score or 0.0), reverse=True
                )
                return merged[:top_n]
            raise
    return _rerank_cross_encoder(query, scored, top_n, model, settings.rerank_max_length)


def _rerank_cross_encoder(
    query: str,
    scored: list[NodeWithScore],
    top_n: int,
    model_name: str,
    max_length: int = 512,
) -> list[NodeWithScore]:
    resolved = str(Path(model_name).resolve()) if Path(model_name).is_dir() else model_name
    texts = [sn.node.get_content() or "" for sn in scored]
    pairs = [(query, t) for t in texts]
    # ⚠️ 必须传 max_length：bge-reranker-v2-m3 默认按模型上限 8192 处理，
    # 长候选会让 cross-encoder 的 O(n²) 注意力爆炸（实测单题 rerank 109s）。
    # 法条节点很短，512 足够判相关性；截断长候选把 rerank 拉回亚秒级。
    raw_scores = _get_cross_encoder(resolved).predict(pairs, max_length=max_length)
    combined = list(zip(scored, raw_scores))
    combined.sort(key=lambda x: float(x[1]), reverse=True)
    out: list[NodeWithScore] = []
    for sn, s in combined[:top_n]:
        out.append(NodeWithScore(node=sn.node, score=float(s)))
    return out
