from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "legal-kb-agent"
    debug: bool = False

    # 本机代理会劫持 localhost（见 src/__init__._apply_local_no_proxy）：
    # 该值在包导入时被并入 os.environ 的 NO_PROXY，httpx / qdrant-client / openai 才看得到。
    no_proxy: str = Field(
        default="localhost,127.0.0.1",
        validation_alias=AliasChoices("NO_PROXY"),
        description="绕过代理的地址列表（本地 Qdrant 必须包含 localhost,127.0.0.1）",
    )

    # 注：DATABASE_URL / db_pool_* / REDIS_* 已移除——持久化改 JSONL（BLUEPRINT D-12 附），
    #     不做 Redis 分布式缓存（第五章「明确不做」）。

    # ── LLM 生成层（D-24：统一到 SenseNova 商汤，废弃智谱 glm 路线）────────
    # 为什么统一：双后端意味着同一份评测可能跑在不同模型上，**指标不可比**；
    # 这正是旧项目"E2 与基线跑在不同版本上"那类事故的翻版。
    # 实测（scripts/bench_llm_backends.py，6 题 p50）：
    #   glm-4.5-flash 32.5s ｜ glm-4-flash 19.4s ｜ sensenova-6.8-flash-lite 4.8s
    # 且商汤的条号可精确到"第（二）项"，与 glm-4.5-flash 同级。
    llm_backend: str = Field(
        default="sensenova",
        validation_alias=AliasChoices("LLM_BACKEND"),
        description="只接受 sensenova；设成别的值会直接报错（不静默回退）",
    )
    llm_model: str = Field(
        default="sensenova-6.8-flash-lite",
        validation_alias=AliasChoices("LLM_MODEL"),
    )
    # 凭据：逗号分隔多个 key，自动轮转；只从环境变量 / .env（已 gitignore）读
    sensenova_api_keys: str = Field(
        default="",
        validation_alias=AliasChoices("SENSENOVA_API_KEYS"),
        description="SenseNova API keys, comma-separated for rotation",
    )

    # 限流与退避（用户要求：确保请求频率，防止频繁 429）
    # 实测 2026-09-28：连续调用约 12 次即报 `429 rpm exhausted`
    # → 光靠"最小间隔"不够，必须按**每分钟请求数**做滑窗限流。
    llm_max_rpm: int = Field(
        default=10, ge=1, le=600,
        validation_alias=AliasChoices("LLM_MAX_RPM"),
        description="每分钟最大请求数（滑窗限流）。实测商汤免费档约 10 rpm 即触顶",
    )
    llm_min_interval_seconds: float = Field(
        default=1.0, ge=0.0,
        validation_alias=AliasChoices("LLM_MIN_INTERVAL_SECONDS"),
        description="两次 LLM 调用之间的最小间隔（全局限流，线程安全）",
    )
    llm_backoff_base_seconds: float = Field(
        default=2.0, ge=0.1,
        validation_alias=AliasChoices("LLM_BACKOFF_BASE_SECONDS"),
        description="429/5xx 退避基数（指数退避：base * 2^attempt）",
    )
    llm_backoff_max_seconds: float = Field(
        default=30.0, ge=1.0,
        validation_alias=AliasChoices("LLM_BACKOFF_MAX_SECONDS"),
    )
    llm_backoff_jitter_seconds: float = Field(
        default=1.0, ge=0.0,
        validation_alias=AliasChoices("LLM_BACKOFF_JITTER_SECONDS"),
        description="退避抖动上限，避免并发请求同时重试再次打爆配额",
    )

    # 注：VLM/OCR（vlm_model）已移除——第五章「图像/扫描件 OCR（云端 VLM 已删）」。

    # Embedding model (sentence-transformers compatible)
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("EMBEDDING_MODEL_NAME"),
        description="sentence-transformers model name or path for embeddings",
    )
    embedding_model_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_MODEL_PATH"),
        description="Override local path for embedding model",
    )
    embedding_dimension: int | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_DIMENSION"),
        description=(
            "期望的 embedding 维度。设置后：模型加载时断言 == 该值，"
            "建库/加载索引时断言 == Qdrant 集合维度，不匹配直接抛错。"
            "旧项目教训：代码默认 1024 而集合是 512 → 应用层 400 而无人发现。"
        ),
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=512,
        validation_alias=AliasChoices("EMBEDDING_BATCH_SIZE"),
        description=(
            "建库时批量编码的 batch size（8 万级节点逐条编码会非常慢）。"
            "实测：batch=32 + fp32 时 GPU 只吃到 74W/220W、单核 100% —— 是 "
            "kernel-launch bound（批次太小、发指令发不过来），8GB 显存下建议 128。"
        ),
    )
    embedding_fp16: bool = Field(
        default=True,
        validation_alias=AliasChoices("EMBEDDING_FP16"),
        description=(
            "CUDA 下用半精度编码。bge-m3 权重 fp32 是 2.27GB，fp16 减半且吞吐约 2 倍；"
            "8GB 显存要同时给 reranker 留位置，默认开。CPU 上自动不生效。"
        ),
    )
    embed_cache_dir: str = Field(
        default="data/.embed_cache",
        validation_alias=AliasChoices("EMBED_CACHE_DIR"),
        description=(
            "Embedding 断点续跑缓存目录（按 模型+维度+精度 分标签）。"
            "空字符串 = 关闭缓存（每次全量编码）。"
        ),
    )
    build_progress_path: str = Field(
        default="data/.embed_cache/progress.json",
        validation_alias=AliasChoices("BUILD_PROGRESS_PATH"),
        description="建库进度落盘路径（done/total/速率/ETA），让『还要多久』可查而不是靠猜",
    )
    reranker_model_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RERANKER_MODEL_PATH"),
        description="Path to reranker model directory on disk",
    )

    # 本地 Embedding / Reranker（torch）：auto=有 NVIDIA CUDA 用 GPU，否则 CPU
    inference_device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="auto | cuda | cpu：与核显/独显切换时建议 auto",
    )

    # 知识库与向量库（相对运行 cwd）
    docs_dir: str = Field(default="data/docs")
    qdrant_collection_name: str = Field(default="rag_kb")
    vector_backend: Literal["qdrant"] = Field(
        default="qdrant",
        validation_alias=AliasChoices("VECTOR_BACKEND"),
        description="向量后端（仅 qdrant 可用）",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL"),
    )
    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY"),
    )
    qdrant_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_PATH"),
        description="本地嵌入式 Qdrant 目录（设此项则不用 Docker / qdrant_url）；未设时默认 data/qdrant_local",
    )

    # 中文知识库 Collection（法律语料为中文单语；见 src/language_router.py）
    qdrant_collection_name_cn: str = Field(
        default="kb_cn_general",
        validation_alias=AliasChoices("QDRANT_COLLECTION_NAME_CN"),
        description="中文法律知识库 Collection",
    )
    docs_dir_cn: str = Field(
        default="data/docs_cn",
        validation_alias=AliasChoices("DOCS_DIR_CN"),
        description="中文知识库文档目录",
    )
    bm25_corpus_path_cn: str = Field(
        default="data/bm25_cn_corpus.jsonl",
        validation_alias=AliasChoices("BM25_CORPUS_PATH_CN"),
        description="中文 BM25 语料路径",
    )

    # 切片：P0 = markdown_heading_overlap；heading_only = 仅按标题不切二次
    chunk_strategy: str = Field(default="hierarchical_recursive")
    chunk_split_on_tiao: bool = Field(
        default=True,
        validation_alias=AliasChoices("CHUNK_SPLIT_ON_TIAO"),
        description=(
            "是否按『条』切分（BLUEPRINT 1.2：法条按条为最小单元）。"
            "关掉则退回只按 markdown 标题切——A 类题目的『条号命中』判分将无法成立，"
            "D 类版本冲突题也失去落点，仅在 D8-4 切块对比实验中作为对照组使用"
        ),
    )
    chunk_size_tokens: int = Field(default=512, ge=64)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    # 检索前 Query Rewrite（LLM）：off=关；on=每次检索必改写；auto=启发式决定是否调用改写
    query_rewrite_mode: Literal["off", "on", "auto"] = Field(default="auto")

    # 混合检索：BM25 + 向量召回合并后再 Rerank（BM25 语料在 reindex 时生成）
    hybrid_bm25_enabled: bool = Field(default=True)
    hybrid_score_normalize: bool = Field(
        default=True,
        validation_alias=AliasChoices("HYBRID_SCORE_NORMALIZE"),
        description="混合召回合并前将向量分与 BM25 分各自 min-max 归一化到 [0,1]，避免 BM25 压过向量",
    )
    hybrid_fusion: Literal["max", "rrf"] = Field(
        default="max",
        validation_alias=AliasChoices("HYBRID_FUSION"),
        description="混合召回融合策略：max=历史分数归一化取最大值；rrf=Reciprocal Rank Fusion",
    )
    hybrid_rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
        validation_alias=AliasChoices("HYBRID_RRF_K"),
        description="RRF 融合公式中的 k：score += 1 / (k + rank)",
    )
    bm25_candidate_top_k: int = Field(default=20, ge=1, le=200)
    bm25_corpus_path: str = Field(default="data/bm25_corpus.jsonl")
    bm25_dict_path: str = Field(
        default="data/dict/legal_terms.txt",
        validation_alias=AliasChoices("BM25_DICT_PATH"),
        description=(
            "jieba 用户词典（法律术语），由 scripts/build_legal_dict.py 从语料构建。"
            "缺省/缺失时退回 jieba 默认词典并记 warning——默认词典会把 "
            "`诉讼时效` 切成 `诉讼`/`时效`，伤 BM25 召回"
        ),
    )

    # 访问控制：有 user_context 时在向量/BM25 检索前按元数据预筛候选 ID（默认开启）
    access_post_filter_safety_net: bool = Field(
        default=True,
        validation_alias=AliasChoices("ACCESS_POST_FILTER_SAFETY_NET"),
        description="True 时在 merge 后再做一次 Post-filter 兜底；正常仅 Pre-filter",
    )

    # 重排序：`qwen3_causal` 用于 Qwen3-Reranker；`cross_encoder` 用于 BGE 等；`auto` 根据本地 config.json 推断
    rerank_enabled: bool = Field(default=True)
    rerank_backend: str = Field(
        default="auto",
        description="auto | qwen3_causal | cross_encoder",
    )
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_max_length: int = Field(
        default=512, ge=64, le=8192,
        description="cross-encoder 重排的 (query,片段) 截断长度。默认 512——bge-reranker-v2-m3 模型上限 8192，"
                    "不截断会让长候选的 O(n²) 注意力爆炸（实测单题 rerank 109s）；法条节点很短，512 足够",
    )
    rerank_candidate_top_k: int = Field(default=20, ge=1, le=100)

    # 检索并行超时（D-14 附三）：一代就是在这里无超时挂死 150s+
    retrieval_parallel_timeout_seconds: float = Field(
        default=20.0, ge=0.0, le=600.0,
        validation_alias=AliasChoices("RETRIEVAL_PARALLEL_TIMEOUT_SECONDS"),
        description="单路检索（向量 / BM25）的超时；超时则该路降级，用另一路结果继续",
    )
    retrieval_trace_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RETRIEVAL_TRACE_ENABLED"),
        description="是否落检索链路 trace（D-15）。关掉可省 IO，但会失去失败归因能力",
    )
    retrieval_total_budget_seconds: float = Field(
        default=45.0, ge=0.0, le=1800.0,
        validation_alias=AliasChoices("RETRIEVAL_TOTAL_BUDGET_SECONDS"),
        description="整条检索链路的总预算；超出即降级返回已有结果，绝不无限等待",
    )
    qwen_rerank_max_length: int = Field(default=8192, ge=512, le=32768)
    qwen_rerank_batch_size: int = Field(default=4, ge=1, le=32)
    qwen_rerank_instruction: str | None = Field(
        default=None,
        description="可选；与 Qwen 官方 Instruct 一致时不填则用默认英文 instruction",
    )

    # 早退门控（生产：语料外查询快速拒答）——向量 top-1 余弦低于阈值就跳过 rerank（见 src/retrieval.py）
    retrieval_early_gate_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("RETRIEVAL_EARLY_GATE_ENABLED"),
        description="True 时：向量 top-1 相似度 < 阈值 → 跳过 rerank 直接返回（后置 similarity gate 会拒），省 off-topic 查询的 rerank 开销",
    )
    retrieval_early_gate_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="早退门控的向量 top-1 余弦阈值；低于此判为语料外、跳过 rerank。"
                    "0.45 由 v1 评测实测校准：同域命中(A–E) 向量 top-1 min=0.588、语料外(F) min=0.386，"
                    "0.45 落在两者间隙——只对明显语料外早退，不误伤同域（留 0.13 余量）",
    )

    # 时效感知重排（D-06，独立可开关；默认关，D4 做开/关对照）
    currency_rerank_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CURRENCY_RERANK_ENABLED"),
        description="True 时在精排后按 status_code 重排：现行有效加权、未生效/废止降权；查询含年份则走时间旅行过滤",
    )
    currency_current_boost: float = Field(default=0.05, ge=0.0, le=1.0, description="status_code=3（现行有效）加分")
    currency_superseded_penalty: float = Field(default=0.10, ge=0.0, le=1.0, description="status_code=2（已被修订）扣分")
    currency_pending_penalty: float = Field(default=0.15, ge=0.0, le=1.0, description="status_code=4（尚未生效）扣分")
    currency_repealed_penalty: float = Field(default=0.25, ge=0.0, le=1.0, description="status_code=1（已废止）扣分")
    currency_status_intent_boost: float = Field(default=0.10, ge=0.0, le=1.0, description="意图感知（D5）：问'生效了吗/时效状态'时给未生效版(status_code=4)加分把它顶上来（这类问题里未生效版正是答案）")

    # 注：领域路由（domain_router_*）与检索意图加权（retrieval_intent_*）已随一代客服残留删除。

    # K2（产品语义）：Rerank 之后的门控；最优重排分低于阈值则拒答（见 src/gates.py）。
    retrieval_gate_enabled: bool = Field(default=True)
    retrieval_similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_score_higher_is_better: bool = Field(
        default=True,
        description="对重排分：True 表示分数越大越相关。False：视为距离类，内部取反后再与阈值比；阈值仍按「越大越好」校准。",
    )
    retrieval_gate_strict_mode: bool = Field(
        default=False,
        description="True 时严格 gate，False 时 relax gate（允许弱匹配通过）",
    )
    retrieval_min_chunks_threshold: int = Field(
        default=1, ge=0, le=10,
        description="至少需要多少个 chunk 才通过 gate",
    )
    refusal_no_results: str = Field(default="知识库中无相关内容")
    refusal_gate_fail: str = Field(default="知识库中无相关内容")

    # 注：behavior_guard / OPA 策略引擎（policy_* / opa_*）/ 多 Agent 图（agent_multi_agent_*）
    #     均属第五章「明确不做（个人项目伪需求，旧项目残留）」，配置已移除。

    agent_grader_mode: str = Field(
        default="auto",
        validation_alias=AliasChoices("AGENT_GRADER_MODE"),
        description="auto | llm | heuristic：Agent grader 模式",
    )
    agent_grader_min_query_overlap: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("AGENT_GRADER_MIN_QUERY_OVERLAP"),
        description="grader 通过的最小 query-chunk n-gram 重叠率",
    )
    agent_max_draft_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        validation_alias=AliasChoices("AGENT_MAX_DRAFT_ATTEMPTS"),
        description="grounding 失败后重试 draft 的最大次数",
    )
    agent_rewrite_use_llm: bool = Field(
        default=True,
        validation_alias=AliasChoices("AGENT_REWRITE_USE_LLM"),
        description="Agent 回环中是否使用 LLM 做 query rewrite",
    )
    grounding_strip_unsupported: bool = Field(
        default=True,
        validation_alias=AliasChoices("GROUNDING_STRIP_UNSUPPORTED"),
        description="grounding 检测后是否自动删除无支撑的句子",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices("LLM_MAX_RETRIES"),
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS"),
    )

    # 注：OpenTelemetry / Langfuse（otel_* / langfuse_*）属第五章「明确不做」；
    #     可观测性走 logging_utils 结构化日志 + telemetry span（D-15 检索链路 trace）。

    # HTTP 鉴权：按 API Key 校验（可选按 IP 白名单）
    api_auth_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("API_AUTH_ENABLED"),
    )
    api_keys: str = Field(
        default="",
        validation_alias=AliasChoices("API_KEYS"),
        description="API 认证密钥，逗号分隔多个 key",
    )
    input_guard_endpoint_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("INPUT_GUARD_ENDPOINT_ENABLED"),
        description="端点级 InputGuard 检查（defense-in-depth，与 HTTP middleware 共存）",
    )
    api_rate_limit_rpm: int = Field(
        default=120,
        ge=0,
        le=10_000,
        validation_alias=AliasChoices("API_RATE_LIMIT_RPM"),
        description="API 速率限制（每分钟请求数），0=不限制",
    )
    api_max_body_bytes: int = Field(
        default=65536,
        ge=0,
        le=10_000_000,
        validation_alias=AliasChoices("API_MAX_BODY_BYTES"),
        description="JSON 请求体最大字节数（Content-Length），0=不限制",
    )

    # 检索结果缓存：仅进程内 L1 精确 LRU（reindex 后 cache_clear）；不做 Redis L2 / 语义 L3（第五章）
    cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("CACHE_ENABLED"),
    )
    cache_max_entries: int = Field(
        default=256,
        ge=8,
        le=10_000,
        validation_alias=AliasChoices("CACHE_MAX_ENTRIES"),
        description="L1 LRU 最大条目数",
    )

    # ── Agent Harness 统一配置 ─────────────────────────────────
    # 注：agent_harness_unified（LangGraph 切换）与 agent_refund_hitl_threshold（一代退款）已移除。
    agent_max_rewrite_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices("AGENT_MAX_REWRITE_ATTEMPTS"),
        description="Harness 评估失败后最大 rewrite 重试次数",
    )
    agent_step_timeout_seconds: float = Field(
        default=30.0,
        ge=5,
        le=120,
        validation_alias=AliasChoices("AGENT_STEP_TIMEOUT_SECONDS"),
        description="Harness 单步工具执行超时（秒）",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
