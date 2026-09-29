"""Qdrant 向量索引：构建、加载（用 qdrant_collection_name 配置）。"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.chunking import build_nodes, load_documents
from src.config_LEGACY_REFERENCE import Settings, get_settings
from src.embed_cache import EmbeddingCache, Progress, cache_tag, text_key
from src.embeddings import get_embedding_model, get_llamaindex_embedding
from src.bm25_store import persist_bm25_corpus

logger = logging.getLogger(__name__)

_index: VectorStoreIndex | None = None
_index_cn: VectorStoreIndex | None = None
_client: QdrantClient | None = None
_client_key: str | None = None
# 上一次 rebuild_index 的分段计时/命中统计（供 scripts/build_index.py 写报告）
last_rebuild_stats: dict = {}


def _qdrant_client(settings: Settings) -> QdrantClient:
    """复用同一 QdrantClient（本地 path 模式不支持多实例并发打开同一目录）。"""
    global _client, _client_key
    key = settings.qdrant_path or settings.qdrant_url
    if _client is not None and _client_key == key:
        return _client
    if settings.qdrant_path and not settings.qdrant_url:
        _client = QdrantClient(path=settings.qdrant_path)
    else:
        kwargs: dict = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    _client_key = key
    return _client


def _collection_vector_size(client: QdrantClient, coll_name: str) -> int | None:
    """读取集合的向量维度；读取失败返回 None（由调用方决定是否致命）。"""
    try:
        info = client.get_collection(coll_name)
        vectors = info.config.params.vectors
    except Exception:
        return None
    if isinstance(vectors, dict):  # 命名向量：取第一个
        for v in vectors.values():
            size = getattr(v, "size", None)
            if size:
                return int(size)
        return None
    size = getattr(vectors, "size", None)
    return int(size) if size else None


def assert_collection_dimension(
    client: QdrantClient, coll_name: str, embed_model, expected: int | None = None
) -> int:
    """断言「embedding 维度 == Qdrant 集合维度」，不一致直接抛错。

    旧项目的教训（HANDOFF §9.2）：代码默认 bge-m3(1024) 而集合是 512
    → 应用层直接 400 而**无人发现**，白跑一轮评测。
    """
    actual = embed_model.assert_dimension(expected)
    coll_dim = _collection_vector_size(client, coll_name)
    if coll_dim is not None and coll_dim != actual:
        raise RuntimeError(
            f"维度不一致：集合 {coll_name!r} 是 {coll_dim} 维，"
            f"而 embedding 模型是 {actual} 维。请重建集合（python scripts/build_index.py --rebuild）"
        )
    return actual


def rebuild_index(
    *, use_cache: bool | None = None, fresh_cache: bool = False
) -> int:
    """写入 Qdrant collection（先 embed 再建库）。

    与原实现的区别（都是“等半天啥都没搞出来”那类问题）：
    1. **维度断言提到编码之前**：原来在 `VectorStoreIndex` 建完之后才核，
       维度错的话 8.8 万次编码白跑。
    2. **向量落盘断点**（`EmbeddingCache`）：崩在最后一步不用重烧编码。
    3. **进度实时写 `progress.json`**：done / total / 速率 / ETA 外部可查。
    """
    global _index, last_rebuild_stats
    settings = get_settings()
    docs_dir = Path(settings.docs_dir).resolve()

    emb_t0 = time.perf_counter()
    documents = load_documents(docs_dir)
    nodes = build_nodes(documents, settings)
    if not nodes:
        logger.warning("无节点可索引: %s", docs_dir)
        return 0

    embed_model = get_embedding_model()
    expected = getattr(settings, "embedding_dimension", None)
    dim = embed_model.assert_dimension(expected)  # ← 编码前就断言，不匹配直接抛错

    batch_size = int(getattr(settings, "embedding_batch_size", 32) or 32)
    fp16 = bool(embed_model.is_sentence_transformers) and bool(
        getattr(settings, "embedding_fp16", False)
    )
    total_nodes = len(nodes)
    prog_path = (getattr(settings, "build_progress_path", "") or "").strip()
    cache_root = (getattr(settings, "embed_cache_dir", "") or "").strip()
    if use_cache is None:
        use_cache = bool(cache_root)

    cache: EmbeddingCache | None = None
    tag = ""
    if use_cache and cache_root:
        tag = cache_tag(embed_model._model_name, dim, fp16)
        cache = EmbeddingCache(Path(cache_root), dim=dim, tag=tag,
                               dtype=np.float16 if fp16 else np.float32)
        if fresh_cache:
            cache.reset()
        else:
            cache.load()

    prog: Progress | None = None
    if prog_path:
        prog = Progress(
            Path(prog_path),
            total=total_nodes,
            meta={
                "model": embed_model._model_name,
                "dim": dim,
                "fp16": fp16,
                "batch_size": batch_size,
                "collection": settings.qdrant_collection_name,
                "cache_tag": tag or "(disabled)",
                "cache_loaded": cache.loaded if cache else 0,
            },
        )

    def _say(line: str) -> None:
        logger.info(line)
        print(line, flush=True)

    todo_idx = [i for i, n in enumerate(nodes) if n.embedding is None]
    _say(
        f"[embed] 待编码 {len(todo_idx)}/{total_nodes} 节点｜batch={batch_size} "
        f"fp16={fp16} dim={dim}｜缓存可用 {cache.loaded if cache else 0} 条"
    )

    # 批量编码：逐条 encode_sync 在 8.8 万节点规模下要跑很久（每次调用都有固定开销），
    # 改成按 batch 走一次前向，实测快一个数量级
    done = 0
    hits = 0
    report_every = max(batch_size, 512)
    for start in range(0, len(todo_idx), batch_size):
        chunk_idx = todo_idx[start : start + batch_size]
        miss_pos: list[int] = []
        miss_texts: list[str] = []
        miss_keys: list[str] = []
        for i in chunk_idx:
            text = nodes[i].get_content(metadata_mode="none") or ""
            key = text_key(text)
            cached = cache.get(key) if cache is not None else None
            if cached is not None:
                nodes[i].embedding = cached
                hits += 1
            else:
                miss_pos.append(i)
                miss_texts.append(text)
                miss_keys.append(key)
        if miss_texts:
            vecs = embed_model.encode_batch_sync(miss_texts, batch_size=len(miss_texts))
            # 注意：只能 zip miss_keys（长度 = miss_pos）。用全量 keys 会错位，
            # 导致“文本 A 的向量存到文本 B 的键下”——缓存命中时静默返回错向量。
            for i, v, key in zip(miss_pos, vecs, miss_keys):
                nodes[i].embedding = v
                if cache is not None:
                    cache.put(key, v)
            if cache is not None:
                cache.flush()
        done += len(chunk_idx)
        if done % report_every < batch_size or done >= len(todo_idx):
            elapsed = time.perf_counter() - emb_t0
            _say(
                f"    embedding {done}/{len(todo_idx)}"
                f" (cache hits {hits})  {elapsed:.0f}s"
            )
        if prog is not None:
            prog.update(done)

    if cache is not None:
        cache.flush(sync=True)
    emb_seconds = time.perf_counter() - emb_t0
    rate = done / emb_seconds if emb_seconds > 0 else 0.0
    _say(
        f"[embed] 完成 {done} 个节点｜耗时 {emb_seconds:.1f}s｜"
        f"{rate:.1f} 节点/s｜缓存命中 {hits}｜新增落盘 {cache.stored if cache else 0}"
    )

    if prog is not None:
        prog.phase = "qdrant_upsert"
        prog.update(done)
    qdrant_t0 = time.perf_counter()
    client = _qdrant_client(settings)
    coll_name = settings.qdrant_collection_name
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    li_embed = get_llamaindex_embedding()
    idx = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=li_embed,
        show_progress=False,
    )
    # 建库后核对集合实际维度（防止 QdrantVectorStore 用了别的维度）
    actual_dim = assert_collection_dimension(client, coll_name, embed_model, expected)
    logger.info("集合 %s 维度核对通过: %s", coll_name, actual_dim)

    if prog is not None:
        prog.phase = "bm25_persist"
        prog.update(done)
    persist_bm25_corpus(nodes, settings)
    _index = idx
    qdrant_seconds = time.perf_counter() - qdrant_t0
    logger.info("Qdrant 索引完成: %s 个节点 -> %s", len(nodes), coll_name)

    if prog is not None:
        prog.phase = "done"
        prog.update(total_nodes)

    last_rebuild_stats = {
        "nodes": len(nodes),
        "nodes_to_embed": len(todo_idx),
        "cache_hits": hits,
        "cache_stored": cache.stored if cache else 0,
        "cache_loaded": cache.loaded if cache else 0,
        "cache_dir": str(cache.dir) if cache else None,
        "batch_size": batch_size,
        "fp16": fp16,
        "embedding_dimension": dim,
        "embedding_seconds": round(emb_seconds, 1),
        "embedding_rate_nodes_per_second": round(rate, 2),
        "index_seconds": round(qdrant_seconds, 1),
    }
    return len(nodes)


def get_vector_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index
    settings = get_settings()
    client = _qdrant_client(settings)
    coll_name = settings.qdrant_collection_name
    try:
        client.get_collection(coll_name)
    except Exception as e:
        raise RuntimeError(
            f"Qdrant 集合 {coll_name!r} 不存在，请先 VECTOR_BACKEND=qdrant python scripts/reindex.py"
        ) from e

    li_embed = get_llamaindex_embedding()
    # 加载索引时核对维度（D2）：集合维度与模型维度不一致 → 检索结果无意义
    assert_collection_dimension(client, coll_name, get_embedding_model(),
                               getattr(settings, "embedding_dimension", None))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=li_embed,
    )
    return _index


def get_vector_index_cn() -> VectorStoreIndex:
    """获取中文知识库向量索引（kb_cn_general Collection）。"""
    global _index_cn
    if _index_cn is not None:
        return _index_cn
    settings = get_settings()
    client = _qdrant_client(settings)
    coll_name = getattr(settings, "qdrant_collection_name_cn", "kb_cn_general")
    try:
        client.get_collection(coll_name)
    except Exception as e:
        raise RuntimeError(
            f"Qdrant 中文集合 {coll_name!r} 不存在，请先运行 python scripts/build_cn_index.py"
        ) from e

    li_embed = get_llamaindex_embedding()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _index_cn = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=li_embed,
    )
    logger.info("中文索引加载完成: %s", coll_name)
    return _index_cn


def clear_index_memory_cache() -> None:
    global _index, _index_cn, _client, _client_key
    _index = None
    _index_cn = None
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _client_key = None


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    return _qdrant_client(settings or get_settings())
