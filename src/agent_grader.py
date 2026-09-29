"""Agent Grader：判定检索证据是否足以回答问题（D-12 附二）。

两条判据（设计见 BLUEPRINT D-12 附二）：
  A. 确定性判据 —— 时效是否匹配意图（`status_code` / 时间旅行）。**D5 接入闭环时补**，
     见下方 `TODO(D5)`。
  B. 语义判据 —— LLM 判定证据充分性；无可用 LLM 时退化为 n-gram 重叠率（可复现、零成本）。

阈值标定：在 **dev 集**上画 precision-recall 曲线取 F1 最高点，在 **test 集**上只报告不回调
（否则构成泄漏，见 docs/02 原则 3）。

注（2026-09-28 D0 修复）：本文件原为迁移时产生的乱码（中文提示词与 docstring 全部变成
``?``），且引用了 Settings 中**不存在**的字段 ``zhipu_chat_model`` / ``zhipuai_api_key``，
导致 LLM 分支恒不执行、`llm_grade` 恒抛 AttributeError → **实际永远退化为 n-gram**。
现改为走项目统一 LLM 入口 ``src.llm.chat_completion``，并把"无 key"交给异常路径兜底，
与 ``agent_grader_mode=auto`` 的语义一致。
"""
from __future__ import annotations

import json
import logging
import re

from src.config_LEGACY_REFERENCE import get_settings

logger = logging.getLogger(__name__)


def ngram_overlap(query: str, text: str, n: int = 3) -> float:
    """字符 n-gram 重叠率：query 的 n-gram 有多少比例出现在 text 中。

    作为 LLM 不可用时的确定性回退判据（可复现、无网络依赖）。
    """
    q_ngrams = {query[i:i + n] for i in range(len(query) - n + 1)}
    if not q_ngrams:
        return 0.0
    t_ngrams = {text[i:i + n] for i in range(len(text) - n + 1)}
    overlap = q_ngrams & t_ngrams
    return len(overlap) / len(q_ngrams)


GRADER_SYSTEM_PROMPT = """你是一个法律问答的证据充分性评估器。判断给定的检索证据是否足以回答用户问题。

只返回 JSON，不要包含任何其他文字：

{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_info": "缺少哪些关键信息；若已充分则为空字符串",
  "rewrite_hint": "若需补充检索，给出改写后的查询；若已充分则为空字符串"
}

判定标准：
- sufficient=true 仅当证据已包含回答问题所需的法条内容或明确规则
- 证据只提到法名但未给出条文内容时，sufficient=false
- 涉及法条时效时，证据必须能说明现行有效版本，否则 sufficient=false
- rewrite_hint 必须是可直接用于检索的查询词，不要写建议性语句"""

_FALLBACK_RESULT = {
    "sufficient": False,
    "confidence": 0.0,
    "missing_info": "",
    "rewrite_hint": "",
}


def llm_grade(query: str, context: str) -> dict:
    """用 LLM 判定证据充分性；失败时返回安全的"不充分"结果。

    调用项目统一入口 ``src.llm.chat_completion``（SenseNova / 智谱由 .env 的
    LLM_BACKEND 决定，见 .env.example）。未配置 key 时抛 RuntimeError → 由调用方回退。
    """
    from src.llm import chat_completion

    user_prompt = f"用户问题：{query}\n\n检索证据：\n{context}\n\n请评估 (JSON)："
    try:
        raw = chat_completion(GRADER_SYSTEM_PROMPT, user_prompt)
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"grader 返回非 dict: {type(data).__name__}")
        return {
            "sufficient": bool(data.get("sufficient", False)),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "missing_info": str(data.get("missing_info", "") or ""),
            "rewrite_hint": str(data.get("rewrite_hint", "") or ""),
        }
    except Exception as exc:
        logger.warning("LLM grader 失败，回退 n-gram：%s", exc)
        return dict(_FALLBACK_RESULT)


def grade_sufficiency(query: str, context: str) -> dict:
    """判定证据是否充分。``agent_grader_mode``：auto（默认）| llm | heuristic。

    - ``llm``：只用 LLM，失败即判不充分
    - ``heuristic``：只用 n-gram，零网络
    - ``auto``：先试 LLM，不可用（无 key / 网络失败）则回退 n-gram

    TODO(D5)：接入判据 A —— 时效是否匹配意图。当前仅有语义判据 B，
    确定性判据在 D5 实现 Agentic 闭环时与本函数合流（BLUEPRINT D-12 附二）。
    """
    settings = get_settings()
    mode = getattr(settings, "agent_grader_mode", "auto")

    if mode in ("llm", "auto"):
        result = llm_grade(query, context)
        if result.get("sufficient") or mode == "llm":
            return result

    # 回退：n-gram 重叠率（确定性、可复现）
    overlap = ngram_overlap(query, context)
    min_overlap = getattr(settings, "agent_grader_min_query_overlap", 0.12)
    sufficient = overlap >= min_overlap
    return {
        "sufficient": sufficient,
        "confidence": min(overlap * 2, 1.0),
        "missing_info": "",
        "rewrite_hint": query if not sufficient else "",
    }
