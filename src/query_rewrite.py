"""检索前 Query Rewrite：口语/指代消解 → 适合向量与 BM25 检索的独立短句。

语言策略：
  中文输入 → 中文输出
  英文输入 → 英文输出
  保持原文语言，不强制翻译
"""
from __future__ import annotations

import logging
import re

from src.config_LEGACY_REFERENCE import Settings
from src.llm import chat_completion
from src.logging_utils import log_structured_event

logger = logging.getLogger(__name__)

_REWRITE_SYS = (
    "你是检索查询改写器。将用户输入改写成一条适合在客服知识库中检索的独立短句。\n"
    "规则：\n"
    "1) 保留核心意图与实体名；\n"
    "2) 去掉寒暄、口语填充词、语气词；\n"
    "3) 不回答用户问题，不解释原因；\n"
    "4) 保持原文语言：中文输入输出中文，英文输入输出英文；\n"
    "5) 只输出一行改写后的检索查询，不加引号或前缀。"
)


# 口语、追问、指代较多时倾向调用改写
_COLLOQUIAL = (
    "吗",
    "呢",
    "嘛",
    "啥",
    "咋",
    "怎么",
    "哪个",
    "帮我",
    "能不能",
    "可以吗",
    "是不是",
    "对不对",
    "好不好",
    "多少啊",
)
# 明显是「短关键词检索」时倾向跳过改写（省一次 LLM）
_KEYWORDISH = re.compile(r"^[\w\u4e00-\u9fff·\-_/\.]{2,40}$")


def should_use_llm_rewrite(user_query: str) -> bool:
    """auto 模式：启发式判断是否有必要调用智谱做检索改写。"""
    s = user_query.strip()
    if len(s) < 3:
        return False
    if any(m in s for m in _COLLOQUIAL):
        return True
    if "？" in s or "?" in s:
        return True
    if any(x in s for x in ("是什么", "什么是", "哪些", "如何", "怎样", "为啥", "为什么")):
        return True
    if len(s) > 42:
        return True
    # 短且无问号、无口语：更像关键词，直接检索
    if len(s) <= 24 and _KEYWORDISH.match(s):
        return False
    return True


def resolve_retrieval_query(
    user_query: str,
    settings: Settings,
    *,
    use_rewrite: bool | None = None,
    trace_id: str | None = None,
) -> str:
    text = user_query.strip()
    if not text:
        return text

    # API 显式关闭 / 打开优先生效
    if use_rewrite is False:
        return text
    force_llm = use_rewrite is True

    mode = settings.query_rewrite_mode

    if not force_llm:
        if mode == "off":
            return text
        if mode == "auto" and not should_use_llm_rewrite(text):
            logger.debug("query rewrite auto: skip rewrite for %r", text[:160])
            return text

    if not settings.sensenova_api_keys:
        if force_llm or mode == "on":
            logger.warning("已启用 query rewrite 但未配置智谱 API Key，跳过改写")
        return text

    try:
        raw = chat_completion(_REWRITE_SYS, f"用户原输入：{text}")
    except Exception:
        logger.exception("query rewrite 调用失败，使用原问题")
        return text

    line = (raw or "").strip().splitlines()[0].strip()
    if len(line) < 2:
        return text
    if len(line) > 512:
        line = line[:512]

    if line != text:
        logger.info(
            "query rewrite: %r -> %r",
            text[:160],
            line[:160],
        )
        log_structured_event(
            trace_id,
            "query_rewrite",
            original_query=text[:500],
            retrieval_query=line[:500],
            mode=mode,
        )
    return line
