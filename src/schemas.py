from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """调用方身份；不传则不做租户/角色过滤。"""

    user_id: str | None = None
    tenant_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    department: str | None = None
    security_clearance: int = Field(
        default=1,
        ge=0,
        le=10,
        description="数值越大权限越高，用于匹配 security_level 文档密级",
    )


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=30)
    use_query_rewrite: bool | None = Field(
        default=None,
        description="是否改写检索句：False=禁用；True=强制智谱改写；None=沿用 query_rewrite_mode",
    )
    user_context: UserContext | None = None
    skip_domain_router: bool = Field(
        default=False,
        description="保留字段以兼容旧调用；领域路由已随一代客服残留移除，当前为 no-op。",
    )


class ChunkHit(BaseModel):
    text: str
    score: float | None = None
    file_path: str | None = None
    file_name: str | None = None
    heading: str | None = None
    node_id: str | None = None
    domain: str | None = Field(
        default=None,
        description="文档 front matter 中的 domain",
    )


class RetrieveResponse(BaseModel):
    query: str
    retrieval_query: str | None = Field(
        default=None,
        description="实际用于检索/重排的查询；与 query 相同时为 null",
    )
    chunks: list[ChunkHit]
    gate_passed: bool = True
    error_code: str | None = None
    behavior: str | None = Field(
        default=None,
        description="behavior guard：human_review 表示未走完整检索",
    )
    refusal_reason_code: str | None = Field(
        default=None,
        description="策略护栏原因码（如 POLICY_*），与 error_code 可一致",
    )
    ranked_quality_scores: list[float] = Field(
        default_factory=list,
        description="门控用的分数降序（rerank 开启时为重排分）",
    )
    router_trace: dict | None = Field(
        default=None,
        description="领域路由：allowed_domains, primary_domain, method, confidence",
    )
    trace_id: str | None = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=30)
    use_query_rewrite: bool | None = Field(default=None, description="同 RetrieveRequest")
    user_context: UserContext | None = None
    skip_domain_router: bool = Field(
        default=False,
        description="同 RetrieveRequest：保留字段，领域路由已移除，当前为 no-op",
    )


class CitationBlock(BaseModel):
    index: int
    file_path: str | None = None
    file_name: str | None = None
    heading: str | None = None
    excerpt: str


class ChatResponse(BaseModel):
    query: str
    retrieval_query: str | None = Field(
        default=None,
        description="实际用于检索的查询；与 query 相同时为 null",
    )
    answer: str
    citations: list[CitationBlock]
    chunks_used: int
    refused: bool = False
    error_code: str | None = None
    behavior: str | None = Field(
        default=None,
        description="normal | human_review：护栏命中时为 human_review",
    )
    refusal_reason_code: str | None = Field(
        default=None,
        description="策略护栏原因码；与 error_code 对齐时可相同",
    )
    ranked_quality_scores: list[float] = Field(
        default_factory=list,
        description="门控分数降序（rerank 开启时为重排分）",
    )
    router_trace: dict | None = None
    trace_id: str | None = None
    citation_overlap_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="答案与引用资料的简易字符重叠度（非严格事实校验）",
    )
    grounding: dict[str, Any] | None = Field(
        default=None,
        description="句级溯源报告（GroundingReport.to_dict）；拒绝/无 chunk 时为 null",
    )
