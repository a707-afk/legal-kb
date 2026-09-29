"""向量索引统一入口：英文 + 中文双 Collection。"""
from __future__ import annotations

from llama_index.core import VectorStoreIndex



def rebuild_index() -> int:
    from src.cache import cache_clear
    from src.vector_store import rebuild_index as _rebuild
    n = _rebuild()
    cache_clear()
    return n


def get_vector_index() -> VectorStoreIndex:
    """获取英文/德文知识库索引 (kb_en_de / rag_kb)。"""
    from src.vector_store import get_vector_index as _get
    return _get()


def get_vector_index_cn() -> VectorStoreIndex:
    """获取中文知识库向量索引 (kb_cn_general)。"""
    from src.vector_store import get_vector_index_cn as _get_cn
    return _get_cn()


def clear_index_memory_cache() -> None:
    from src.vector_store import clear_index_memory_cache as _clear
    _clear()
