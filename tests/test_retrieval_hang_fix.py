"""回归防护：retrieve_scored_nodes 挂死修复（docs/FIX-retrieval-hang-20260927.md）。

修复前症状：首次调用先烧 ~65s CPU，随后阻塞 >5 分钟不返回（取证调用栈：
embedding 模型在查询路径懒加载时 SentenceTransformer 对 huggingface.co 发
HEAD 版本检查，无外网时每次 TCP connect ~21s（WinError 10060）× 每文件重试
5 次指数退避 × 多个探测文件）。

本测试断言：
1. embedding 模型能解析到本地 HF 快照（修复前提）；
2. 冷启动（含模型加载 + BM25 缓存命中）≤ 60s（挂死场景 >150s）；
3. 清空检索缓存后的二次完整检索 ≤ 10s；
4. BM25 持久缓存二次加载 ≤ 5s；
5. 返回 top_k=5 个节点。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# 复现环境默认值（已在环境中显式设置时以环境为准；必须在 import app 之前）
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("QDRANT_URL", "http://localhost:18333")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "legal_kb")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
os.environ.setdefault("INFERENCE_DEVICE", "cuda")
os.environ.setdefault("RERANK_ENABLED", "false")
os.environ.setdefault("DOMAIN_ROUTER_ENABLED", "false")
os.environ.setdefault("QUERY_REWRITE_MODE", "off")
os.environ.setdefault("HYBRID_BM25_ENABLED", "true")

import pytest  # noqa: E402

from src.retrieval import retrieve_scored_nodes  # noqa: E402

QUERY = "劳动合同试用期最长不能超过多久"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BM25_CORPUS = PROJECT_ROOT / "data" / "bm25_corpus.jsonl"


@pytest.fixture(scope="module")
def index_and_settings():
    from src.config_LEGACY_REFERENCE import get_settings
    from src.vector_index import get_vector_index

    return get_vector_index(), get_settings()


def test_embedding_model_resolves_local_snapshot():
    """模型必须能解析到本地 HF 快照（否则查询路径会访问 huggingface.co → 挂死）。"""
    from src.embeddings import resolve_local_hf_snapshot

    model_name = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    snapshot = resolve_local_hf_snapshot(model_name)
    if snapshot is None:
        pytest.skip(f"本地 HF 缓存中无 {model_name} 快照（修复前提不满足）")
    assert Path(snapshot).is_dir()


def test_retrieve_cold_no_hang_hot_under_10s_and_returns_5_nodes(index_and_settings):
    """核心回归：同一查询二次调用 ≤10s 且返回 5 个节点；冷启动 ≤60s（原挂死 >5min）。"""
    index, settings = index_and_settings
    from src.cache import cache_clear

    # 冷启动（embedding 模型懒加载 + BM25 缓存加载），有界断言：≤60s
    t0 = time.monotonic()
    first = retrieve_scored_nodes(
        index, QUERY, 5, settings, skip_domain_router=True
    )
    cold = time.monotonic() - t0
    assert len(first.nodes) == 5, f"首次调用应返回 5 个节点，实际 {len(first.nodes)}"
    assert cold <= 60.0, f"冷启动 {cold:.1f}s 超过 60s（挂死场景 faulthandler 150s 未返回）"

    # 清空 L1/L2 缓存，强制二次调用走完整检索（模型已热、BM25 缓存已持久化）
    cache_clear()
    t1 = time.monotonic()
    second = retrieve_scored_nodes(
        index, QUERY, 5, settings, skip_domain_router=True
    )
    hot = time.monotonic() - t1
    assert len(second.nodes) == 5, f"二次调用应返回 5 个节点，实际 {len(second.nodes)}"
    assert hot <= 10.0, f"二次检索 {hot:.1f}s 超过 10s"
    print(f"\n[cost] cold={cold:.2f}s hot={hot:.2f}s")


def test_bm25_persist_cache_second_load_under_5s():
    """BM25 持久缓存：二次加载必须 ≤5s（原实现每进程全量 jieba 分词，分钟级）。"""
    from src.bm25_store import _load_corpus_disk, clear_bm25_memory_cache

    if not BM25_CORPUS.is_file():
        pytest.skip(f"BM25 语料缺失: {BM25_CORPUS}")

    clear_bm25_memory_cache()
    t0 = time.monotonic()
    _, ids, _ = _load_corpus_disk(str(BM25_CORPUS))
    first_load = time.monotonic() - t0
    assert ids, "BM25 语料应包含文档"

    clear_bm25_memory_cache()  # 模拟新进程：只走磁盘缓存
    t1 = time.monotonic()
    _, ids, _ = _load_corpus_disk(str(BM25_CORPUS))
    second_load = time.monotonic() - t1
    assert ids, "缓存二次加载应包含文档"
    assert second_load <= 5.0, (
        f"BM25 缓存二次加载 {second_load:.2f}s 超过 5s"
        f"（首次 {first_load:.1f}s，缓存可能未生效）"
    )
    print(f"\n[cost] bm25 first_load={first_load:.2f}s second_load={second_load:.2f}s")
