"""法务审查 Agent —— 核心包。

模块地图（迁移自旧仓库 REACTAGENT，D2 阶段逐个裁剪 import）：

检索链路
    chunking      有界切块 + 分层节点（父/子）
    embeddings    本地 HF 快照加载（零网络）
    bm25_store    BM25 索引 + pickle 持久化
    vector_store  Qdrant 集合读写
    vector_index  索引入口
    rerank        bge-reranker 精排
    gates         拒答门控

检索编排
    retrieval     混合检索：向量 ∥ BM25 → 归一化/RRF 融合 → 重排 → 门控
    access_prefilter  检索前权限过滤（pre-filter）
    access_control    权限规则与 audience 映射
    query_rewrite     查询改写（启发式 + LLM）

生成与校验
    llm                 LLM 客户端（D2 改为智谱）
    citation            句级 grounding + 条号级引用校验
    logging_utils       结构化日志（检索 trace 的基础）

Agent 层
    agent.harness        ReAct Harness：规划 / 执行 / 评估 / 改写回环 / HITL
    agent.state          运行状态定义
    agent.tool_registry  工具注册表（副作用 / 风险 / 幂等 / 参数校验）
    agent.research_tools local_search + synthesize
    agent.permission_gate 工具权限门
    audit_service        审计落盘

API
    api.routes_rag     /retrieve /chat
    api.routes_agent   /agent/run

基础设施（一代残留清理后新建的轻量本地实现）
    cache              进程内检索结果 LRU（不做 Redis）
    telemetry          轻量 span 追踪（D-15 trace 接入点，不做 OTel）
    language_router    语言路由（收敛为中文单语）
    input_sanitizer    轻量输入/输出护栏（InputGuard/OutputGuard，不做 OPA）
    db                 AgentRun/AgentStep 的 JSONL 落盘（D-12 附，替代 Postgres）

待建（D5/D6）
    currency           时效核查（本项目特有，尚不存在）
    trace              检索链路 JSONL trace + 回放（replay_trace.py）

✅ 当前状态：命名空间已统一到 `src.*`，import 断链清零、全模块可 import
（跑 scripts/audit_imports.py 复核）。运行检索/Agent 仍需 Qdrant + 本地模型 + .env（见 docs/06）。
"""
from __future__ import annotations

import os as _os


def _apply_local_no_proxy() -> None:
    """把本地回环地址并入 ``NO_PROXY``，避免系统代理劫持 localhost。

    实测 2026-09-29：httpx 默认 ``trust_env=True``，在 Windows 上会经
    ``urllib.request.getproxies()`` 从注册表读到系统代理（``http://127.0.0.1:10809``），
    于是连 ``http://localhost:18333``（Qdrant）也被塞进代理转发 → ``ReadTimeout``
    → 检索全部返回空 → Agent 每题都降级成「未检索到」，评测指标整体失真。

    `.env` 里虽然写了 ``NO_PROXY=localhost,127.0.0.1``，但 pydantic-settings 只把值
    读进 Settings 对象、**不会写回 os.environ**，而 httpx 只认 os.environ，
    所以那条配置一直是死配置。这里在包导入时补齐，并保留外部已设的值。
    """
    local = ["localhost", "127.0.0.1", "::1"]
    configured = ""
    try:
        from src.config_LEGACY_REFERENCE import get_settings

        configured = str(getattr(get_settings(), "no_proxy", "") or "")
    except Exception:  # 配置不可用时也要兜住本地回环
        configured = ""
    if not configured:
        configured = _os.environ.get("NO_PROXY") or _os.environ.get("no_proxy") or ""
    merged = [p.strip() for p in f"{configured},{','.join(local)}".split(",") if p.strip()]
    value = ",".join(dict.fromkeys(merged))
    _os.environ["NO_PROXY"] = value
    _os.environ["no_proxy"] = value


_apply_local_no_proxy()
