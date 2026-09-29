"""混合召回分数归一化与合并。"""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import (
    _merge_hybrid_by_node_id,
    _merge_hybrid_by_rrf,
    _normalize_scores_minmax,
)


def _node(nid: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text="t", id_=nid), score=score)


def test_normalize_minmax_spreads_scores():
    nodes = [_node("a", 10.0), _node("b", 20.0)]
    out = _normalize_scores_minmax(nodes)
    assert out[0].score == 0.0
    assert out[1].score == 1.0


def test_merge_takes_max_normalized_per_id():
    vec = [_node("x", 0.9), _node("y", 0.1)]
    bm25 = [_node("x", 5.0), _node("z", 100.0)]
    merged = _merge_hybrid_by_node_id(vec, bm25, normalize_scores=True)
    by_id = {sn.node.node_id: float(sn.score) for sn in merged}
    assert "x" in by_id
    assert "y" in by_id
    assert "z" in by_id
    assert by_id["x"] == 1.0
    assert by_id["z"] == 1.0
    assert merged[0].score >= merged[-1].score


def test_rrf_boosts_documents_seen_by_both_retrievers():
    vec = [_node("x", 0.9), _node("y", 0.8)]
    bm25 = [_node("z", 100.0), _node("x", 5.0)]
    merged = _merge_hybrid_by_rrf(vec, bm25, k=60)
    assert merged[0].node.node_id == "x"
    by_id = {sn.node.node_id: float(sn.score) for sn in merged}
    assert by_id["x"] > by_id["y"]
    assert by_id["x"] > by_id["z"]


def test_merge_can_use_rrf_fusion():
    vec = [_node("x", 0.9), _node("y", 0.8)]
    bm25 = [_node("z", 100.0), _node("x", 5.0)]
    merged = _merge_hybrid_by_node_id(vec, bm25, fusion="rrf", rrf_k=60)
    assert merged[0].node.node_id == "x"
