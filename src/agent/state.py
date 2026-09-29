"""Legal KB Agent state (TypedDict).

Tracks one agent run: the user question, the multi-step plan, working
memory of gathered evidence, and the final synthesized answer with
citations.
"""
from __future__ import annotations

from typing import Any, TypedDict


class ResearchAgentState(TypedDict, total=False):
    """State carried through the ReAct loop.

    Fields are grouped by concern. All are optional (total=False) so the
    loop can populate them incrementally.
    """

    # ── Run identity ────────────────────────────────────────────────
    run_id: str
    session_id: str | None              # multi-turn session binding
    trace_id: str | None                # observability trace id

    # ── User objective ──────────────────────────────────────────
    objective: str                      # the user question
    user_context: dict[str, Any]        # tenant_id, roles, scopes
    risk_level: str                     # low | medium | high | critical

    # ── Decomposition (planning) ────────────────────────────────────
    sub_questions: list[str]            # objective broken into sub-questions
    current_subquestion: str | None     # which sub-question we're working on now
    subquestion_index: int              # pointer into sub_questions

    # ── ReAct loop control ──────────────────────────────────────────
    iterations: int                     # current ReAct step
    max_iterations: int                 # budget cap (default 8)
    thoughts: list[dict[str, Any]]      # per-step LLM reasoning trace
    actions: list[dict[str, Any]]       # per-step tool calls
    observations: list[dict[str, Any]]  # per-step tool results

    # ── Working memory (current sub-question) ───────────────────────
    gathered_facts: list[dict[str, Any]]      # facts found this sub-question
    gathered_sources: list[dict[str, Any]]    # sources (url/doc_id + snippet)
    sufficient: bool                          # has reflect decided "enough"?

    # ── Long-term memory (cross sub-questions) ──────────────────────
    verified_facts: list[dict[str, Any]]      # deduped, prioritized
    all_sources: list[dict[str, Any]]

    # ── Synthesis / output ──────────────────────────────────────────
    draft_answer: str | None
    final_answer: str | None
    citations: list[dict[str, Any]]           # [{index, source_id, snippet}]
    faithfulness_score: float | None          # grounding metric
    coverage_score: float | None              # fact coverage metric

    # ── HITL ────────────────────────────────────────────────────────
    human_review_required: bool
    final_action: str                         # completed | waiting_approval | failed

    # ── Audit ───────────────────────────────────────────────────────
    audit_trace: list[dict[str, Any]]         # step-by-step for replay
    errors: list[dict[str, Any]]
