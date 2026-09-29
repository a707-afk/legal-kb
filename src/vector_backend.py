"""向量后端解析：当前仅支持 qdrant。"""
from __future__ import annotations

from typing import Literal

ResolvedBackend = Literal["qdrant"]


def resolve_vector_backend(settings=None) -> Literal["qdrant"]:
    """解析向量后端。本项目只使用 qdrant（其他后端属第五章「明确不做」）。"""
    return "qdrant"
