"""生成答案与引用 chunk 的可溯源校验（字符 overlap + 句级 grounding）。"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 句级 grounding 默认阈值（n-gram / embedding 两套，可通过参数覆盖）
DEFAULT_NGRAM_SUPPORT_THRESHOLD = 0.28
DEFAULT_EMBEDDING_SUPPORT_THRESHOLD = 0.42
DEFAULT_MAX_UNSUPPORTED_RATE = 0.35
_MIN_SENTENCE_LEN = 2

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")

# markdown 结构行（不承载事实主张）
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_MD_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# 行内装饰符：引用号 [1] 与 markdown 强调/表格/引用符，去掉后更利于与纯文本证据做 n-gram
_MD_INLINE = re.compile(r"\[\d+\]|[*`>|]|^\s*(?:[-+]\s+|\d+[.)]\s+)")

# 免责声明/格式套话：非事实主张，不计入 grounding 分母
_BOILERPLATE = ("仅供参考", "不构成法律意见", "不构成任何法律意见", "不构成专业法律意见")


def _is_boilerplate(sentence: str) -> bool:
    s = (sentence or "").strip()
    return bool(s) and any(m in s for m in _BOILERPLATE)


def _strip_markdown_scaffolding(text: str) -> str:
    """去掉 markdown 结构行与行内装饰，只留下承载事实的句子。

    为什么需要：句级 grounding 判的是「主张是否有据」。标题（`## 一、适用版本`）、
    水平线（`---`）、表格分隔行（`|---|---|`）本身不含事实，若计入 unsupported，
    结构化答案会被一律判成「不可溯源」——实测 D 类长答案 unsupported_rate
    0.40~0.67，grounding_rate 被压到 0.0（分母是全部句子）。
    表格数据行同理：`| 条号 | 第一条 |` 这种单元格拼不成可 n-gram 对齐的句子，
    一律计入会把「表格化表述」误判成幻觉。
    剔除结构行后，分母变成真正的陈述句，指标才反映「答案有没有编」。
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if (_MD_HEADING.match(s) or _MD_RULE.match(s)
                or _MD_TABLE_SEP.match(s) or _MD_TABLE_ROW.match(s)):
            continue
        out.append(_MD_INLINE.sub(" ", s))
    return "\n".join(out)


@dataclass
class SentenceGrounding:
    """单句与证据 chunk 的对齐结果。"""

    sentence: str
    support_score: float
    best_chunk_id: str | None
    unsupported: bool
    method: str = "ngram"


@dataclass
class GroundingReport:
    """答案句级溯源报告。"""

    sentences: list[SentenceGrounding] = field(default_factory=list)
    overlap_ratio: float = 1.0
    unsupported_sentence_rate: float = 0.0
    passed: bool = True
    method: str = "ngram"
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "method": self.method,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "unsupported_sentence_rate": round(self.unsupported_sentence_rate, 4),
            "feedback": self.feedback,
            "sentences": [
                {
                    "sentence": s.sentence,
                    "support_score": round(s.support_score, 4),
                    "best_chunk_id": s.best_chunk_id,
                    "unsupported": s.unsupported,
                    "method": s.method,
                }
                for s in self.sentences
            ],
        }


def citation_overlap_ratio(answer: str, chunk_texts: list[str]) -> float:
    """返回 [0,1]：引用中是否有足够长的片段出现在答案中。"""
    a = re.sub(r"\s+", "", (answer or "").strip())
    if len(a) < 8:
        return 1.0
    hits = 0
    for raw in chunk_texts:
        t = re.sub(r"\s+", "", (raw or "").strip())
        if len(t) < 12:
            continue
        ok = False
        for win in range(min(24, len(t)), 11, -1):
            for i in range(0, len(t) - win + 1):
                frag = t[i : i + win]
                if frag in a:
                    ok = True
                    break
            if ok:
                break
        if ok:
            hits += 1
    if not chunk_texts:
        return 1.0
    return min(1.0, hits / min(3, len(chunk_texts)))


def _split_sentences(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(raw) if p.strip()]
    out = [p for p in parts if len(p) >= _MIN_SENTENCE_LEN]
    if not out and raw:
        out = [raw]
    return out


def _char_ngrams(s: str, n: int = 3) -> set[str]:
    s = re.sub(r"\s+", "", s)
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    ga = _char_ngrams(a, n)
    gb = _char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def _substring_boost(sentence: str, chunk_text: str) -> float:
    """短子串命中时对 n-gram 分数加权。"""
    s = re.sub(r"\s+", "", sentence)
    c = re.sub(r"\s+", "", chunk_text)
    if len(s) < 4 or not c:
        return 0.0
    best = 0.0
    for win in range(min(16, len(s)), 3, -1):
        for i in range(0, len(s) - win + 1):
            frag = s[i : i + win]
            if frag in c:
                best = max(best, 0.45 + 0.55 * (win / max(len(s), 1)))
                break
        if best >= 0.9:
            break
    return best


def _ngram_support_score(sentence: str, chunk_text: str) -> float:
    base = _ngram_jaccard(sentence, chunk_text)
    boost = _substring_boost(sentence, chunk_text)
    return max(base, boost)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _chunk_id(chunk: dict[str, Any], index: int) -> str:
    cid = chunk.get("node_id") or chunk.get("chunk_id")
    if cid:
        return str(cid)
    return f"chunk-{index}"


def _embedding_sentence_scores(
    sentences: list[str],
    chunks: list[dict[str, Any]],
) -> list[tuple[float, str | None]] | None:
    """尝试用现有 embedding 模型算句-chunk 余弦；失败返回 None。"""
    if not sentences or not chunks:
        return []
    try:
        from src.embeddings import get_embedding_model

        model = get_embedding_model()
        chunk_texts = [str(c.get("text") or "") for c in chunks]
        chunk_ids = [_chunk_id(c, i) for i, c in enumerate(chunks)]

        sent_vecs = [model.encode_sync(s) for s in sentences]
        chunk_vecs = [model.encode_sync(t) for t in chunk_texts]

        results: list[tuple[float, str | None]] = []
        for sv in sent_vecs:
            best_score = 0.0
            best_id: str | None = None
            for cv, cid in zip(chunk_vecs, chunk_ids):
                sc = _cosine(sv, cv)
                if sc > best_score:
                    best_score = sc
                    best_id = cid
            results.append((best_score, best_id))
        return results
    except Exception as e:
        logger.debug("embedding grounding 不可用，回退 n-gram: %s", e)
        return None


def sentence_level_grounding(
    answer: str,
    chunks_with_meta: list[dict[str, Any]],
    *,
    support_threshold: float | None = None,
    max_unsupported_rate: float = DEFAULT_MAX_UNSUPPORTED_RATE,
    prefer_embedding: bool = True,
    strip_scaffolding: bool = True,
) -> GroundingReport:
    """句级 grounding：每句找最佳 chunk，标记 unsupported 并汇总通过率。

    ``strip_scaffolding=True`` 时先剔除 markdown 结构行（标题/分隔线/表格线），
    它们不承载事实主张，计入分母会系统性压低通过率（见 `_strip_markdown_scaffolding`）。
    """
    chunk_texts = [str(c.get("text") or "") for c in chunks_with_meta]
    overlap = citation_overlap_ratio(answer, chunk_texts)
    body = _strip_markdown_scaffolding(answer) if strip_scaffolding else answer
    # 若整篇都是结构行（无实质陈述），退回原文，避免"空句集"被当成通过
    if strip_scaffolding and not body.strip() and (answer or "").strip():
        body = answer
    sentences = _split_sentences(body)
    # 免责声明等套话不承载事实，剔除出分母（否则短答案被套话拖到 33%+ 而误判）
    sentences = [s for s in sentences if not _is_boilerplate(s)]

    if not sentences:
        return GroundingReport(
            sentences=[],
            overlap_ratio=overlap,
            unsupported_sentence_rate=0.0,
            passed=bool((answer or "").strip()),
            method="ngram",
            feedback="无有效句子" if not (answer or "").strip() else "ok",
        )

    if not chunks_with_meta:
        sg = [
            SentenceGrounding(
                sentence=s,
                support_score=0.0,
                best_chunk_id=None,
                unsupported=True,
                method="ngram",
            )
            for s in sentences
        ]
        return GroundingReport(
            sentences=sg,
            overlap_ratio=overlap,
            unsupported_sentence_rate=1.0,
            passed=False,
            method="ngram",
            feedback="无证据 chunk，全部句子 unsupported",
        )

    method = "ngram"
    embed_scores = _embedding_sentence_scores(sentences, chunks_with_meta) if prefer_embedding else None
    if embed_scores is not None:
        method = "embedding"
        thr = support_threshold if support_threshold is not None else DEFAULT_EMBEDDING_SUPPORT_THRESHOLD
    else:
        thr = support_threshold if support_threshold is not None else DEFAULT_NGRAM_SUPPORT_THRESHOLD

    grounded: list[SentenceGrounding] = []
    for i, sent in enumerate(sentences):
        if embed_scores is not None:
            best_score, best_id = embed_scores[i]
        else:
            best_score = 0.0
            best_id: str | None = None
            for j, chunk in enumerate(chunks_with_meta):
                sc = _ngram_support_score(sent, str(chunk.get("text") or ""))
                if sc > best_score:
                    best_score = sc
                    best_id = _chunk_id(chunk, j)
        grounded.append(
            SentenceGrounding(
                sentence=sent,
                support_score=best_score,
                best_chunk_id=best_id,
                unsupported=best_score < thr,
                method=method,
            )
        )

    unsupported_n = sum(1 for g in grounded if g.unsupported)
    rate = unsupported_n / len(grounded) if grounded else 0.0
    passed = rate <= max_unsupported_rate

    if passed:
        feedback = f"grounding_ok ({method}, unsupported_rate={rate:.2f})"
    else:
        bad = [g.sentence[:40] for g in grounded if g.unsupported][:3]
        feedback = f"unsupported_rate={rate:.2f} ({method}); 例: {' | '.join(bad)}"

    return GroundingReport(
        sentences=grounded,
        overlap_ratio=overlap,
        unsupported_sentence_rate=rate,
        passed=passed,
        method=method,
        feedback=feedback,
    )


def verify_grounding(
    answer: str,
    evidence_text: str,
    threshold: float = DEFAULT_NGRAM_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """检查 answer 中每句话是否被 evidence_text 支撑，返回 unsupported 句子列表。"""
    if not answer or not evidence_text:
        return {"unsupported_sentences": [], "passed": True, "unsupported_rate": 0.0}

    chunks_with_meta = [{"text": evidence_text, "chunk_id": "evidence-0"}]
    report = sentence_level_grounding(
        answer,
        chunks_with_meta,
        support_threshold=threshold,
        prefer_embedding=False,
        max_unsupported_rate=1.0,
    )
    unsupported_sentences = [sg.sentence for sg in report.sentences if sg.unsupported]
    return {
        "unsupported_sentences": unsupported_sentences,
        "passed": report.passed,
        "unsupported_rate": report.unsupported_sentence_rate,
    }


def strip_unsupported_sentences(text: str, context: str, threshold: float = 0.28) -> str:
    if not text or not context:
        return text
    result = verify_grounding(text, context)
    unsupported = result.get("unsupported_sentences", [])
    if not unsupported:
        return text
    for s in unsupported:
        text = text.replace(s, "").strip()
    return text or "(content filtered by grounding)"
