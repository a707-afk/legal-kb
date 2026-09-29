# -*- coding: utf-8 -*-
"""从旧仓库 rag-kb-project(REACTAGENT) 迁移可用资产到本仓库。

设计原则
--------
1. **只复制，不移动、不删除。** 旧仓库完整保留，可随时 diff / 回滚。
2. **按"是否解决过真问题"取舍**，不按"代码看起来好不好"取舍。
3. **闭包完整**：迁进来的模块，其依赖也要一起迁（或明确标为"待重写"），
   否则会出现"迁了模块但 import 全断"的假迁移。
4. **迁移必须包含测试**。只迁实现不迁测试 = 把"已验证"变成"未验证"。
5. 原始数据全量迁（含全部历史版本），因为路线 A（法条版本冲突）需要它们。
6. 结果落 `reports/migration-manifest.json`，可审计。

版本
----
v2（2026-09-28）：补上 v1 漏掉的 tests/、环境文件（docker-compose / requirements /
Dockerfile / pytest.ini / CI）、以及 7 个被依赖的 app 模块。并清理 v1 放错位置的旧脚本。

用法
----
    python scripts/migrate_from_legacy.py --dry-run   # 只看要做什么
    python scripts/migrate_from_legacy.py             # 实际执行（幂等）
    python scripts/migrate_from_legacy.py --audit     # 只做完整性审计，不复制
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _env_path(name: str, default: str | None = None) -> Path:
    raw = os.environ.get(name) or default
    if not raw:
        raise SystemExit(
            f"需要设置环境变量 {name}（旧仓库的绝对路径）。\n"
            f"迁移是一次性动作，路径由环境变量提供，不写死在代码里。"
        )
    return Path(raw).expanduser()


# 旧仓库位置：从环境变量读取，避免把本机绝对路径写进代码
SRC_REPO = _env_path("LEGACY_REPO")
# 旧仓库上一级目录（存放 raw_legal_dl / _data_tools，不在旧仓库内）
LEGACY_DATA_ROOT = _env_path("LEGACY_DATA_ROOT", str(SRC_REPO.parent))
# 目标仓库：脚本所在目录的上一级
DST = Path(__file__).resolve().parent.parent

# ============================================================ 代码（实现）
CODE_CARRY = [
    # --- v1 ---
    ("app/chunking.py", "src/chunking.py",
     "有界切块 + 分层节点；修复过 MemoryError，有 27 个测试覆盖"),
    ("app/embeddings.py", "src/embeddings.py",
     "resolve_local_hf_snapshot：本地 HF 快照零网络加载，修掉首次检索数分钟挂死"),
    ("app/bm25_store.py", "src/bm25_store.py",
     "BM25 索引 pickle 持久化，冷建 50.9s → 二次 0.9s"),
    ("app/qdrant_index_store.py", "src/vector_store.py", "Qdrant 建库/加载"),
    ("app/retrieval_pipeline.py", "src/retrieval.py",
     "混合检索：min-max 与 RRF 两套融合、分库路由、检索门控"),
    ("app/rerank.py", "src/rerank.py", "重排封装"),
    ("app/citation_verify.py", "src/citation.py", "引用/溯源校验——幻觉治理核心"),
    ("app/inference_device.py", "src/inference_device.py", "auto/cuda/cpu 设备选择"),
    ("app/schemas.py", "src/schemas.py", "请求/响应模型（需裁剪电商字段）"),
    ("app/agent/harness.py", "src/agent/harness.py", "自研 ReAct Harness——执行型能力载体"),
    ("app/agent/research_tools.py", "src/agent/research_tools.py", "local_search + synthesize 工具"),
    ("app/agent/tool_registry.py", "src/agent/tool_registry.py", "工具注册表"),
    ("app/agent/permission_gate.py", "src/agent/permission_gate.py", "工具权限门"),
    ("app/routes_rag.py", "src/api/routes_rag.py", "问答路由（需去 SenseNova/多租户依赖）"),
    # --- v2 补迁：被上述模块 import 的闭包 ---
    ("app/agent/state.py", "src/agent/state.py",
     "【v2 补】Harness 的状态定义，harness.py 依赖"),
    ("app/vector_index.py", "src/vector_index.py",
     "【v2 补】向量索引入口（rebuild_index / get_vector_index），多处依赖"),
    ("app/retrieval_gates.py", "src/gates.py",
     "【v2 补】拒答门控 evaluate_similarity_gate——D-09 的核心，81 行"),
    ("app/query_rewrite.py", "src/query_rewrite.py",
     "【v2 补】查询改写，E6 实验需要，125 行"),
    ("app/llm.py", "src/llm.py",
     "【v2 补】LLM 客户端 + 多 key 轮询，143 行（需改为智谱）"),
    ("app/observability.py", "src/logging_utils.py",
     "【v2 补】log_structured_event——**这是结构化日志，不是 Prometheus**，"
     "是 D-17 检索链路 trace 的基础（v1 误删）"),
    ("app/sse.py", "src/sse.py", "【v2 补】流式输出，P2 阶段用"),
    # --- v3 补迁：import 审计发现的真依赖 ---
    ("app/access_prefilter.py", "src/access_prefilter.py",
     "【v3 补】检索前权限过滤——**D-17 保留的能力**，且 `src/retrieval.py` 已接入调用点"
     "（line 262/288/362），不迁则检索直接崩"),
    ("app/access_control.py", "src/access_control.py",
     "【v3 补】权限规则与 audience 映射，被 access_prefilter 依赖，146 行"),
    ("app/qwen_rerank.py", "src/qwen_rerank.py",
     "【v3 补】`src/rerank.py` 依赖它，187 行"),
    ("app/services/audit_service.py", "src/audit_service.py",
     "【v3 补】`src/agent/permission_gate.py` 依赖它，90 行"),
    # --- v4 补迁：逐文件审计发现的真漏 ---
    ("app/routes_agent.py", "src/api/routes_agent.py",
     "【v4 补】`/agent/ticket` + **`/agent/run`** 两个端点——**Harness 的对外 API 入口**，"
     "D5 的 Agentic 闭环需要这个入口，140 行"),
    ("scripts/d3_forensics_hang.py", "scripts/legacy/d3_forensics_hang.py",
     "【v4 补】挂死取证脚本（faulthandler 全线程栈 + psutil 采样区分「烧 CPU」与「纯阻塞」）——"
     "方法论可复用，99 行"),
    ("scripts/smoke_agent_ticket.py", "eval/legacy-v3/scripts/smoke_agent_ticket.py",
     "【v4 补】工单冒烟测试，作参考，129 行"),
    # --- 参考，必须重写 ---
    ("app/config.py", "src/config_LEGACY_REFERENCE.py",
     "⚠️ **只作参考，必须重写**。479 行里大半是已删模块的配置；"
     "且它的默认值（embedding=BAAI/bge-m3）与集合维度不一致，是旧项目故障源之一"),
]

# ============================================================ 测试
TESTS_CARRY = [
    ("tests/conftest.py", "tests/conftest.py", "测试基础设施，必须"),
    ("tests/test_chunking_guard.py", "tests/test_chunking_guard.py",
     "【关键】有界切块回归保护（MemoryError 修复的守卫）"),
    ("tests/test_retrieval_hang_fix.py", "tests/test_retrieval_hang_fix.py",
     "【关键】挂死修复回归保护（二次调用 ≤10s 断言）"),
    ("tests/test_hybrid_merge.py", "tests/test_hybrid_merge.py", "【关键】混合合并（max/RRF）"),
    ("tests/test_citation_verify.py", "tests/test_citation_verify.py", "【关键】引用校验"),
    ("tests/test_access_prefilter.py", "tests/test_access_prefilter.py",
     "【关键】权限 pre-filter——D-17 保留的能力"),
    ("tests/test_access_prefilter_security.py", "tests/test_access_prefilter_security.py",
     "【关键】权限安全边界"),
    ("tests/test_retrieval_intent_boost.py", "tests/test_retrieval_intent_boost.py", "检索意图加权"),
    ("tests/test_domain_router_embedding_mock.py", "tests/test_domain_router_embedding_mock.py", "分库路由"),
    ("tests/test_grounding_strip.py", "tests/test_grounding_strip.py", "grounding 剥离"),
    ("tests/test_llm_zhipu.py", "tests/test_llm_zhipu.py", "【关键】一代就用了智谱，可直接参考"),
    ("tests/test_llm_async.py", "tests/test_llm_async.py", "LLM 异步调用"),
    ("tests/test_harness_run.py", "tests/test_harness_run.py", "Harness 端到端"),
    ("tests/agent/test_agent_harness.py", "tests/agent/test_agent_harness.py", "Harness 单测"),
]

TESTS_DROP = [
    ("tests/test_cache.py", "对应模块 cache.py 已删（Redis）"),
    ("tests/test_db_async.py", "对应模块 db/ 已删"),
    ("tests/test_degradation.py", "对应模块 degradation.py 已删"),
    ("tests/test_metrics_counters.py", "对应模块 metrics.py 已删（Prometheus）"),
    ("tests/test_metrics_endpoint.py", "同上"),
    ("tests/test_sse_routes.py", "SSE 路由属 P2，暂不测"),
    ("tests/test_api_guard.py", "对应模块 api_guard.py 已删"),
    ("tests/test_health_ready.py", "健康检查属部署层，暂不做"),
    ("tests/api/**", "旧业务 API（documents/jobs/tickets）已删"),
    ("tests/db/**", "db 层已删"),
    ("tests/eval/**", "旧评测 runner 已删，评测体系重做"),
    ("tests/ingestion/**", "旧 ingestion pipeline 已删，导入重写"),
    ("tests/worker/**", "Celery worker 已删"),
    ("tests/audit_retrieval_cn.py / bench.py / e2e_validation.py / smoke_server.py", "工具脚本，非测试"),
]

# ============================================================ 环境与工程文件
ENV_CARRY = [
    ("docker-compose.yml", "docker-compose.yml",
     "【v2 补】起 Qdrant/Redis/Postgres；需裁剪掉已删服务"),
    ("Dockerfile", "Dockerfile", "【v2 补】镜像定义"),
    ("pytest.ini", "pytest.ini", "【v2 补】pytest 配置（pythonpath/testpaths/asyncio_mode）"),
    (".dockerignore", ".dockerignore", "【v2 补】"),
    ("requirements-torch-cuda.txt", "requirements-torch-cuda.txt", "【v2 补】torch CUDA 版本约束"),
    ("requirements.txt", "requirements-LEGACY.txt",
     "⚠️ 【v2 补】只作参考，必须重写：含 redis/arq/sqlalchemy/alembic/prometheus 等已删依赖"),
    (".github/workflows/lint.yml", ".github/workflows/lint.yml", "【v2 补】CI lint"),
    (".github/workflows/test.yml", ".github/workflows/test.yml",
     "【v2 补】CI 测试——D8-3 评测闸门的起点"),
]

# ============================================================ 原始数据
DATA_CARRY = [
    ("raw_legal_dl/npc", "data/raw/npc",
     "flk 官方原始 docx（法律 734 / 行政法规 851 / 司法解释 881）+ metadata.jsonl"
     "（含 status_code / effective_date，路线 A 的 GT 来源）"),
    ("_data_tools/LeCaRD-main/data/label", "data/raw/lecard/label", "LeCaRD 学术金标"),
    ("_data_tools/LeCaRD-main/data/query", "data/raw/lecard/query", "LeCaRD 107 个查询案例"),
    ("_data_tools/LeCaRD-main/data/corpus", "data/raw/lecard/corpus", "LeCaRD 语料索引"),
    ("_data_tools/LeCaRD-main/data/others", "data/raw/lecard/others",
     "罪名表 / 停用词——可用于 BM25 法律术语词典"),
    ("_data_tools/LeCaRD-main/data/candidates", "data/raw/lecard/candidates",
     "⚠️ 566MB，含真实姓名 PII。**第一轮不纳入评测**，仅为将来做类案检索保留（不入 git）"),
    ("_data_tools/LawBench-main/data/zero_shot/1-1.json", "data/raw/lawbench/1-1.json",
     "法条内容问答 500 条"),
    ("_data_tools/LawBench-main/data/zero_shot/1-2.json", "data/raw/lawbench/1-2.json",
     "备用扩容源"),
]

# ============================================================ 旧产出（只读参考）
LEGACY_REF = [
    ("data/eval/legal/eval_set_v1.jsonl", "eval/legacy-v3/eval_set_v1.jsonl",
     "v3 评测集 148 条——**已知同源缺陷，仅作设计参考，不得复用**"),
    ("data/eval/legal/EVAL.md", "eval/legacy-v3/EVAL.md", "旧评测报告（数字互相矛盾）"),
    ("data/eval/legal/baseline_v0_summary.json", "eval/legacy-v3/baseline_v0_summary.json", "旧基线"),
    ("data/eval/legal/baseline_e2_rerank_summary.json", "eval/legacy-v3/baseline_e2_rerank_summary.json", "旧 E2（与基线不可比）"),
    ("data/eval/legal/case_include.json", "eval/legacy-v3/case_include.json", "案例库金标对齐清单"),
    ("docs/REVIEW-SESSION-LOG-20260927.md", "docs/archive/REVIEW-SESSION-LOG-20260927.md", "执行层审查报告"),
    ("docs/STRATEGY-REVIEW-20260928.md", "docs/archive/STRATEGY-REVIEW-20260928.md", "战略层审查报告"),
    ("docs/FIX-retrieval-hang-20260927.md", "docs/archive/FIX-retrieval-hang-20260927.md", "挂死修复取证报告"),
    # 旧脚本归到 legacy-v3/scripts/（v2 修正 v1 的位置错误）
    ("scripts/ingest_legal_corpus.py", "eval/legacy-v3/scripts/ingest_legal_corpus.py",
     "旧导入脚本——含字段名 bug（status vs status_code）与单一版本策略"),
    ("scripts/d3_build_eval_set.py", "eval/legacy-v3/scripts/d3_build_eval_set.py", "旧评测集构建脚本"),
    ("scripts/d4_baseline.py", "eval/legacy-v3/scripts/d4_baseline.py", "旧基线脚本（recall@5 口径混用）"),
    ("scripts/d4_e2_rerank_eval.py", "eval/legacy-v3/scripts/d4_e2_rerank_eval.py", "旧重排脚本（含 abs() bug）"),
    ("scripts/verify_chunking_bounded.py", "scripts/verify_chunking_bounded.py",
     "切块有界性验证——**仍然有效，继续用**"),
]

# ============================================================ 明确不迁
CODE_DROP = [
    ("app/vlm.py", "云端 VLM/OCR 调用，项目不需要图像理解"),
    ("app/ingestion/**", "旧 ingestion pipeline（823 行），导入逻辑重写"),
    ("app/policy/**", "OPA 策略即代码引擎（572 行），个人项目伪需求"),
    ("app/supervisor/**", "监督者 Agent，属多 Agent 实体，蓝图明确不做"),
    ("app/worker/**", "Celery 任务队列，无异步批处理需求"),
    ("app/db/**", "PostgreSQL + Alembic，检索型项目无业务库需求"),
    ("app/multi_tenant.py", "多租户管理，个人项目伪需求（只保留隔离语义，见 D-17）"),
    ("app/access_control.py / access_prefilter.py",
     "实现不迁；但其**测试已迁**（用于验证 pre-filter 有效性），"
     "能力由 src/gates.py + payload filter 承接"),
    ("app/redis_client.py / cache.py", "Redis 依赖；单机项目用进程内 LRU"),
    ("app/metrics.py", "Prometheus 指标——对 RAG 失败归因无用（见 docs/08）"),
    ("app/degradation.py / input_sanitizer.py / api_guard.py", "电商时代的守卫逻辑"),
    ("app/api/**", "旧业务 API（documents/approvals/tickets）"),
    ("app/main.py / logging_config.py / qwen_rerank.py / qwen_embedding*",
     "应用装配层，随 config 一起重写；qwen_rerank 与 rerank.py 重复"),
    ("scripts/run_eval_*.py / seed_*.py / bench_server_models.py / rebuild_index_server.py",
     "电商时代的评测与运维脚本"),
    ("frontend/", "电商时代的 React UI（66MB 含 node_modules），UI 重做"),
    ("static/ / alembic/ / models/", "前端产物 / 数据库迁移 / 空模型目录"),
    ("data/bm25_cs_corpus.jsonl", "61MB 旧电商语料"),
    ("data/eval_agent_ticket.jsonl", "旧工单评测"),
    ("data/docs_research/ / data/docs/", "旧语料目录（空或已迁移）"),
    ("sitecustomize.py", "全局 WMI 补丁，影响本机所有 Python 进程"),
]


def human(p: Path) -> str:
    if p.is_file():
        return f"{p.stat().st_size / 1024:.0f} KB"
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{n / 1024 / 1024:.1f} MB"


def copy_item(src: Path, dst: Path, dry: bool) -> str:
    if not src.exists():
        return "MISSING"
    if dst.exists():
        return "skip(exists)"
    if dry:
        return f"would copy ({human(src)})"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache"))
    else:
        shutil.copy2(src, dst)
    return f"copied ({human(dst)})"


def resolve(rel: str) -> Path:
    """raw_legal_dl 与 _data_tools 在旧仓库的上一级目录（不在仓库内），其余在仓库内。"""
    if rel.startswith("_data_tools") or rel.startswith("raw_legal_dl"):
        return LEGACY_DATA_ROOT / rel
    return SRC_REPO / rel


def run_group(title: str, items, dry: bool, key: str, manifest: dict) -> tuple:
    print("=" * 74)
    print(title)
    n = 0
    for rel_src, rel_dst, why in items:
        st = copy_item(resolve(rel_src), DST / rel_dst, dry)
        if st.startswith("copied"):
            n += 1
        manifest.setdefault(key, []).append(
            {"src": rel_src, "dst": rel_dst, "why": why, "status": st})
        print(f"  {st:24s} {rel_src:46s} -> {rel_dst}")
    return n, len(items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", action="store_true", help="只审计，不复制")
    args = ap.parse_args()
    dry = args.dry_run or args.audit

    manifest: dict = {}
    n1, t1 = run_group("【1/5】代码（实现）", CODE_CARRY, dry, "code", manifest)
    n2, t2 = run_group("【2/5】测试（v2 补：只迁实现不迁测试 = 假迁移）", TESTS_CARRY, dry, "tests", manifest)
    n3, t3 = run_group("【3/5】环境与工程文件（v2 补）", ENV_CARRY, dry, "env", manifest)
    n4, t4 = run_group("【4/5】原始数据（体量大，含多版本）", DATA_CARRY, dry, "data", manifest)
    n5, t5 = run_group("【5/5】旧产出 → 只读参考", LEGACY_REF, dry, "legacy_ref", manifest)

    for rel, why in TESTS_DROP:
        manifest.setdefault("tests_dropped", []).append({"path": rel, "why": why})
    for rel, why in CODE_DROP:
        manifest.setdefault("dropped", []).append({"path": rel, "why": why})

    print("=" * 74)
    if not dry:
        out = DST / "reports" / "migration-manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"迁移清单已写入: {out}")
    else:
        print("(dry-run / audit：未写任何文件)")

    print(f"\n汇总：代码 {n1}/{t1} ｜ 测试 {n2}/{t2} ｜ 环境 {n3}/{t3} ｜ "
          f"数据 {n4}/{t4} ｜ 参考 {t5} ｜ 不迁：测试 {len(TESTS_DROP)} 类 + 代码 {len(CODE_DROP)} 类")
    missing = [m["src"] for g in ("code", "tests", "env", "data", "legacy_ref")
               for m in manifest.get(g, []) if m["status"] == "MISSING"]
    if missing:
        print(f"\n⚠️ 源文件缺失（需人工确认）：{missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
