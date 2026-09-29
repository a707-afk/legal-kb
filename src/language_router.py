"""语言路由：本项目语料为**中文单语**（法律 / 行政法规 / 司法解释 / FAQ）。

一代客服项目残留了「中英德多语种分库路由」；法律 KB 只有一个中文 collection，
因此这里把多语种收敛为单语：
- detect_language：含 CJK 即判 zh（法律查询恒为中文），否则 other；
- get_collection_for_lang：恒路由到中文集合。
保留这层薄封装是为了不打散 retrieval.py 既有的「语言→集合」结构，同时去掉多库伪需求。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class LangRoute:
    """语言路由结果：语言 + 目标 collection 名。"""

    lang: str
    collection_name: str


def detect_language(text: str) -> str:
    """含中文字符判为 "zh"，否则 "other"（法律语料实际恒为 zh）。"""
    if text and _CJK.search(text):
        return "zh"
    return "other"


def get_collection_for_lang(lang: str, settings: Any) -> LangRoute:
    """中文单语单集合：恒返回主集合（build_index 建的就是 qdrant_collection_name）。"""
    name = getattr(settings, "qdrant_collection_name", None) or "legal_kb"
    return LangRoute(lang="zh", collection_name=name)
