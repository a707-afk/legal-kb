"""检索门控：K2 指「Rerank 之后的门控」，对重排分数做阈值判断（非向量召回排名第 2 的 chunk）。"""
from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

from src.config_LEGACY_REFERENCE import Settings
from src.logging_utils import log_structured_event


@dataclass
class GateResult:
    passed: bool
    error_code: str | None
    ranked_scores: list[float]  # 质量降序（第一名最好），与门控用的是同一套分数


def _scores_quality_descending(
    scored: list[NodeWithScore], settings: Settings
) -> list[float]:
    raw: list[float] = []
    for sn in scored:
        if sn.score is None:
            continue
        raw.append(float(sn.score))
    if not raw:
        return []
    if settings.retrieval_score_higher_is_better:
        return sorted(raw, reverse=True)
    mx = max(raw)
    inverted = [mx - x for x in raw]
    return sorted(inverted, reverse=True)


def evaluate_similarity_gate(
    scored: list[NodeWithScore],
    settings: Settings,
    *,
    trace_id: str | None = None,
) -> GateResult:
    """在 **重排后** 的 ``scored`` 上评估：取最优分（第一名）与阈值比较；低于阈值则拒答。"""
    if not settings.retrieval_gate_enabled:
        rg = _scores_quality_descending(scored, settings)
        result = GateResult(True, None, rg)
        log_structured_event(
            trace_id,
            "gate",
            passed=result.passed,
            error_code=result.error_code,
            best_score=rg[0] if rg else None,
            gate_disabled=True,
        )
        return result

    ranked = _scores_quality_descending(scored, settings)
    if not ranked:
        result = GateResult(False, "NO_SCORE", ranked)
        log_structured_event(
            trace_id,
            "gate",
            passed=result.passed,
            error_code=result.error_code,
            best_score=None,
        )
        return result

    thr = settings.retrieval_similarity_threshold
    best = ranked[0]
    if best >= thr:
        result = GateResult(True, None, ranked)
    else:
        result = GateResult(False, "SIMILARITY_GATE_FAIL", ranked)
    log_structured_event(
        trace_id,
        "gate",
        passed=result.passed,
        error_code=result.error_code,
        best_score=best,
    )
    return result
