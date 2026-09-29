"""时效感知重排（D-06，独立可开关模块）。

职责：在语义检索/精排之后，按法条时效元数据（status_code / effective_date）调整最终排序。
- **默认**（查询未指定时间）：现行有效(status_code=3)加权；已修订(2)轻降；未生效(4)/已废止(1)降权。
- **时间旅行**（查询含年份锚点，如"2020 年时"）：同一法名只保留在该时点已生效
  (effective_date <= 该年)的最新一版。

设计约束（见 why/03）：必须是**独立、可单独开关**的一步——否则无法把"提升来自时效"
从"提升来自检索/精排"里分离出来，D4 的"开/关对照 + 过期法条率"就无从归因。
默认 `currency_rerank_enabled=False`，让基线保持纯语义排序；评测时显式打开做对照。
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 查询里的年份锚点："2020年" / "2020 年时" / "1998年"
_YEAR_RE = re.compile(r"(?:19|20)\d{2}(?=\s*年)")


def extract_year_anchor(query: str) -> int | None:
    """从查询抽时间锚点年份（"2020年时怎么规定"→2020）；无则 None。"""
    m = _YEAR_RE.search(query or "")
    return int(m.group(0)) if m else None


# 意图感知（D5）：区分"问内容"与"问时效状态本身"。
# 问"《X》现在生效了吗/什么时候施行"时，未生效版(status_code=4)正是答案，绝不能像默认模式那样降权。
# 模式必须足够特指（带"了吗/是否/什么时候"），否则会误伤 D-version_current 的"现行有效的规定是什么"（那是问内容）。
_STATUS_INTENT_PATTERNS = (
    "生效了吗", "是否生效", "有没有生效", "尚未生效", "何时生效", "什么时候生效",
    "什么时候施行", "何时施行", "生效日期", "施行日期", "废止了吗", "是否废止",
    "失效了吗", "是否失效", "还有效吗", "是否有效", "是否现行有效", "现行有效吗",
    "时效状态", "有没有效",
)


def detect_status_intent(query: str) -> bool:
    """查询是否在问"时效/生效状态"本身（而非问法条内容）。"""
    q = query or ""
    return any(p in q for p in _STATUS_INTENT_PATTERNS)


def _meta(sn: Any) -> dict:
    return getattr(sn.node, "metadata", None) or {}


def _status_of(sn: Any) -> int | None:
    sc = _meta(sn).get("status_code")
    try:
        return int(sc) if sc is not None else None
    except (TypeError, ValueError):
        return None


def _score(sn: Any) -> float:
    try:
        return float(sn.score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def apply_currency_rerank(
    nodes: list,
    query: str,
    settings: Any,
    trace: Any | None = None,
) -> list:
    """按时效重排/过滤候选。关闭或空输入时原样返回。"""
    if not getattr(settings, "currency_rerank_enabled", False) or not nodes:
        return nodes

    year = extract_year_anchor(query)
    if year is not None:
        out = _time_travel_filter(nodes, year)
        if trace is not None:
            trace.record("currency", mode="time_travel", year=year,
                         in_count=len(nodes), out_count=len(out))
        return out

    # 意图感知（D5）：问"生效了吗/时效状态" → 把未生效版(4)顶上来（这正是问题所问），其余中立不降权
    if detect_status_intent(query):
        pend_boost = float(getattr(settings, "currency_status_intent_boost", 0.10))
        adj: list[tuple[float, int, Any]] = []
        for i, sn in enumerate(nodes):
            delta = pend_boost if _status_of(sn) == 4 else 0.0
            adj.append((_score(sn) + delta, i, sn))
        adj.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        out = [sn for _, _, sn in adj]
        if trace is not None:
            trace.record("currency", mode="status_intent", in_count=len(nodes),
                         pending_boosted=sum(1 for sn in nodes if _status_of(sn) == 4))
        return out

    boost = float(getattr(settings, "currency_current_boost", 0.05))
    pen_superseded = float(getattr(settings, "currency_superseded_penalty", 0.10))
    pen_pending = float(getattr(settings, "currency_pending_penalty", 0.15))
    pen_repealed = float(getattr(settings, "currency_repealed_penalty", 0.25))

    adjusted: list[tuple[float, int, Any]] = []
    for i, sn in enumerate(nodes):
        sc = _status_of(sn)
        if sc == 3:
            delta = boost
        elif sc == 2:
            delta = -pen_superseded
        elif sc == 4:
            delta = -pen_pending
        elif sc == 1:
            delta = -pen_repealed
        else:
            delta = 0.0
        adjusted.append((_score(sn) + delta, i, sn))
    # 稳定排序：调整后分数降序；同分保持原相对顺序（靠原下标 i）
    adjusted.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    out = [sn for _, _, sn in adjusted]

    if trace is not None:
        trace.record("currency", mode="reweight", in_count=len(nodes),
                     current_in_topk=sum(1 for sn in nodes if _status_of(sn) == 3))
    return out


def _time_travel_filter(nodes: list, year: int) -> list:
    """同一法名只保留 effective_date <= year-12-31 的最新一版；无则保留分最高的一版。"""
    cutoff = f"{year}-12-31"
    by_title: dict[str, list] = {}
    order: list[str] = []
    for sn in nodes:
        title = str(_meta(sn).get("title") or _meta(sn).get("file_name") or id(sn))
        if title not in by_title:
            by_title[title] = []
            order.append(title)
        by_title[title].append(sn)

    kept: list = []
    for title in order:
        group = by_title[title]
        eligible = [sn for sn in group
                    if str(_meta(sn).get("effective_date") or "") and
                    str(_meta(sn).get("effective_date")) <= cutoff]
        if eligible:
            kept.append(max(eligible, key=lambda sn: str(_meta(sn).get("effective_date"))))
        else:
            kept.append(max(group, key=_score))
    kept.sort(key=_score, reverse=True)
    return kept
