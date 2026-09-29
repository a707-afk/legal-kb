"""Agent API — unified Harness only（/agent/run）。

一代客服的 /agent/ticket（customer_id / customer_tier / 工单）已随残留移除；
D-11「冲突核查任务单」将在 D5 基于 /agent/run 的 Harness 闭环重建。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.agent.harness import run_agent_harness

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


# ── Agent Run API ────────────────────────────────────────────────


class AgentRunRequest(BaseModel):
    objective: str = Field(..., description="Task objective / user query")
    tenant_id: str = Field(default="default")
    user_id: str = Field(default="anonymous")
    session_id: str | None = None
    budget: dict | None = None


class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    final_answer: str | None = None
    final_action: str | None = None
    human_review_required: bool = False
    total_steps: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    tool_error_count: int = 0
    permission_deny_count: int = 0
    errors: list[dict] = []
    approvals: list[dict] = []


@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest, request: Request):
    """Execute an agent run through the full harness: plan -> execute -> evaluate -> HITL."""
    tenant_id = getattr(request.state, "tenant_id", None) or req.tenant_id

    result = await run_agent_harness(
        objective=req.objective,
        tenant_id=tenant_id,
        user_id=req.user_id,
        session_id=req.session_id,
        budget=req.budget,
    )

    return AgentRunResponse(
        run_id=result.run_id,
        status=result.status,
        final_answer=result.final_answer,
        final_action=result.final_action,
        human_review_required=result.human_review_required,
        total_steps=result.total_steps,
        total_tool_calls=result.total_tool_calls,
        total_latency_ms=result.total_latency_ms,
        tool_error_count=result.tool_error_count,
        permission_deny_count=result.permission_deny_count,
        errors=result.errors,
        approvals=result.approvals,
    )
