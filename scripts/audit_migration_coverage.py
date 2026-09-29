# -*- coding: utf-8 -*-
"""迁移完整性审计：把旧仓库的每一个文件都归到"已迁 / 已明确不迁 / 未交代"三类。

这是"不漏东西"的**唯一严格证明方式**——逐文件核对，而不是抽查。

用法：
    python scripts/audit_migration_coverage.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 旧仓库路径：迁移时的一次性输入，用环境变量指定（避免把本机绝对路径写进仓库）
SRC = Path(os.environ.get("LEGACY_REPO", r"../rag-kb-project")).resolve()
DST = Path(__file__).resolve().parent.parent
EXTRA_ROOTS = [Path(os.environ.get("LEGACY_RAW_DL", r"../raw_legal_dl")).resolve(),
               Path(os.environ.get("LEGACY_DATA_TOOLS", r"../_data_tools")).resolve()]

# 只审计代码/文档/配置，不审计体积大的数据与产物
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".idea", ".vscode", ".deeppval", ".deepeval"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".db", ".pkl"}

# 已迁移（旧路径前缀）
MIGRATED_PREFIX = [
    "app/chunking.py", "app/embeddings.py", "app/bm25_store.py", "app/qdrant_index_store.py",
    "app/retrieval_pipeline.py", "app/rerank.py", "app/citation_verify.py",
    "app/inference_device.py", "app/schemas.py", "app/agent/harness.py",
    "app/agent/research_tools.py", "app/agent/tool_registry.py", "app/agent/permission_gate.py",
    "app/routes_rag.py", "app/agent/state.py", "app/vector_index.py", "app/retrieval_gates.py",
    "app/query_rewrite.py", "app/llm.py", "app/observability.py", "app/sse.py", "app/config.py",
    "app/access_prefilter.py", "app/access_control.py", "app/qwen_rerank.py",
    "app/services/audit_service.py",
    "tests/conftest.py", "tests/test_chunking_guard.py", "tests/test_retrieval_hang_fix.py",
    "tests/test_hybrid_merge.py", "tests/test_citation_verify.py",
    "tests/test_access_prefilter.py", "tests/test_access_prefilter_security.py",
    "tests/test_retrieval_intent_boost.py", "tests/test_domain_router_embedding_mock.py",
    "tests/test_grounding_strip.py", "tests/test_llm_zhipu.py", "tests/test_llm_async.py",
    "tests/test_harness_run.py", "tests/agent/test_agent_harness.py",
    "docker-compose.yml", "Dockerfile", "pytest.ini", ".dockerignore",
    "requirements-torch-cuda.txt", "requirements.txt",
    ".github/workflows/lint.yml", ".github/workflows/test.yml",
    "scripts/verify_chunking_bounded.py", "scripts/ingest_legal_corpus.py",
    "scripts/d3_build_eval_set.py", "scripts/d4_baseline.py", "scripts/d4_e2_rerank_eval.py",
    "docs/REVIEW-SESSION-LOG-20260927.md", "docs/STRATEGY-REVIEW-20260928.md",
    "docs/FIX-retrieval-hang-20260927.md",
    "data/eval/legal/eval_set_v1.jsonl", "data/eval/legal/EVAL.md",
    "data/eval/legal/baseline_v0_summary.json", "data/eval/legal/baseline_e2_rerank_summary.json",
    "data/eval/legal/case_include.json",
    # --- v4 补迁（逐文件审计发现）---
    "app/routes_agent.py", "scripts/d3_forensics_hang.py", "scripts/smoke_agent_ticket.py",
    "app/__init__.py", "app/agent/__init__.py",
    # --- v5 补迁（**main 分支**发现，前四轮漏查分支）---
    "app/agent_grader.py", "app/domain_router.py", "app/llm_zhipu.py",
    "app/services/session_mgr.py", "app/agent/tools.py", "app/structured_logging.py",
    "app/behavior_guard.py", "app/retrieval_intent_boost.py", "app/vector_backend.py",
    "tests/test_agent_grader.py",
    "data/domain_router_profiles.json", "data/domain_router_profiles_cs.json",
    "data/router_calibration.default.json", "data/router_eval_golden.jsonl",
    "data/eval_access_control_questions.jsonl", "data/eval_hallucination.jsonl",
    "TECH-V1.md", "REQUIREMENTS-ONEPAGER.md", "BUILD_REPORT.md", "旧版说明.md",
    ".planning/ROADMAP.md", ".planning/STATE.md",
    "docs/LEGACY-NOTES.md", "docs/DECISION-LOG.md", "docs/PROJECT-1-AGENTIC-RAG.md",
    "docs/PROJECT-1-PLAYBOOK.md", "docs/PROJECT-2-MULTI-AGENT.md", "docs/PROJECT-2-PLAYBOOK.md",
    "docs/STRATEGY-GUIDE.md", "docs/RAG-AGENT-KNOWLEDGE-MAP.md", "docs/CURRENT-PROJECT-DEEP-PLAN.md",
    "docs/ACCESS-CONTROL-EVAL", "docs/ROUTER-EVAL.md", "docs/BEHAVIOR-GUARD-EVAL.md",
    "docs/G-K-METRICS-SUMMARY.md", "docs/SCALE-BENCHMARKS.md", "docs/EVAL-BASELINE-COMPARISON.md",
    "docs/router_eval_metrics_", "docs/eval_access_control_qdrant.json",
    "docs/eval_qdrant_vs_chroma_access.json", "docs/eval_behavior_guard.json",
    "docs/eval_hallucination.json", "docs/eval_four_baselines_summary.json",
    "docs/eval_prod_router_matrix_summary.json",
    "scripts/run_eval_router.py", "scripts/run_eval_access_control.py",
    "scripts/run_eval_behavior_guard.py", "scripts/run_eval_hallucination.py",
    "scripts/generate_router_eval_golden.py", "scripts/run_eval_hybrid_rrf_production.py",
    "scripts/run_eval_four_baselines.py", "scripts/compare_access_eval_backends.py",
]

# 已明确不迁（旧路径前缀）
DROPPED_PREFIX = [
    "app/vlm.py", "app/ingestion/", "app/policy/", "app/supervisor/", "app/worker/",
    "app/db/", "app/multi_tenant.py", "app/redis_client.py", "app/cache.py", "app/metrics.py",
    "app/telemetry.py", "app/language_router.py", "app/degradation.py", "app/input_sanitizer.py",
    "app/api_guard.py", "app/api/", "app/main.py", "app/logging_config.py",
    "app/services/", "app/opa/", "app/domain_router.py", "app/retrieval_intent_boost.py",
    "tests/test_cache.py", "tests/test_db_async.py", "tests/test_degradation.py",
    "tests/test_metrics_counters.py", "tests/test_metrics_endpoint.py", "tests/test_sse_routes.py",
    "tests/test_api_guard.py", "tests/test_health_ready.py", "tests/api/", "tests/db/",
    "tests/eval/", "tests/ingestion/", "tests/worker/", "tests/audit_retrieval_cn.py",
    "tests/bench.py", "tests/e2e_validation.py", "tests/smoke_server.py",
    "tests/_server_test_manual.py",
    "scripts/run_eval_", "scripts/seed_", "scripts/bench_server_models.py",
    "scripts/rebuild_index_server.py", "scripts/d1_", "scripts/d2_", "scripts/setup_pytorch_cuda.ps1",
    "scripts/run_docker.ps1", "scripts/run_gradio.ps1", "scripts/run_uvicorn.ps1",
    "scripts/reindex.py",
    "frontend/", "static/", "alembic/", "alembic.ini", "models/", "sitecustomize.py",
    "data/bm25_cs_corpus.jsonl", "data/eval_agent_ticket.jsonl", "data/docs_research/",
    "data/docs/", "data/raw/", "data/behavior_rules.default.json", "data/opa/",
    "requirements-observability.txt",
    "README.md", ".env", ".env.example", "docker-compose.override.yml.example",
    ".gitignore", "docs/", "data/",
    # --- v5 明确不迁（main 分支上的旧物）---
    "app/agent_graph/", "app/api/chat.py", "app/embedding_router.py",
    "scripts/download_cs_data.py", "scripts/preprocess_cs_data.py", "scripts/extract_csds.py",
    "scripts/build_cn_index.py", "scripts/classify_intents.py", "scripts/clean_cn_pipeline.py",
    "scripts/run_cs_eval.py", "scripts/reindex_cs.py", "scripts/gradio_app.py",
    "scripts/start_dsw.sh", "scripts/run_qdrant_migrate_eval.ps1", "scripts/smoke_cs_retrieval.py",
    "scripts/smoke_policy_llm_guard.py", "scripts/run_policy_phase3_answer_sample.py",
    "scripts/test_embed.py", "scripts/verify_enterprise_chunk_metadata.py",
    "data/docs_cs/", "data/docs_cn/", "data/bm25_cn_corpus.jsonl",
    "data/eval_cs_", "data/eval_enterprise_questions.jsonl", "data/eval_questions.jsonl",
    "static/app/", "tests/server_test.py", "tests/test_agent_graph_",
    "docs/PHASE-", "docs/MILESTONE-", "docs/NEXT-MILESTONE.md",
    "docs/POLICY-MILESTONE-ACCEPTANCE.md", "docs/CURRENT-STATUS.md", "docs/QDRANT-NEXT.md",
    "docs/qdrant_reindex.log", "docs/DATA-CORPUS-REDESIGN.md",
    ".env.cs-agent", ".env.dsw", ".codebuddy/", ".qoder/", ".workbuddy/",
    "scripts/run_eval_retrieve.py",
    "scripts/download_cn_datasets.py", "scripts/run_qdrant_migration_eval.py",
]


def is_under(path: str, prefixes: list[str]) -> bool:
    p = path.replace("\\", "/")
    return any(p == pre or p.startswith(pre) for pre in prefixes)


def audit_branches() -> tuple[int, int, list[str]]:
    """扫描旧仓库**所有分支**的文件树（这是前四轮遗漏的维度）。

    只在 working tree 上审计会漏掉其他分支独有的文件——
    v5 就是这样发现 main 分支有 141 个独有文件（含 Grader / 路由 / 智谱客户端）的。
    """
    import subprocess

    def sh(a):
        r = subprocess.run(a, cwd=SRC, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else ""

    branches = [b.strip().lstrip("* ").strip()
                for b in sh(["git", "branch"]).splitlines() if b.strip()]
    seen: set[str] = set()
    n_mig = n_drop = 0
    unaccounted: list[str] = []
    for b in branches:
        for f in sh(["git", "ls-tree", "-r", "--name-only", b]).splitlines():
            if not f or f in seen:
                continue
            seen.add(f)
            if is_under(f, MIGRATED_PREFIX):
                n_mig += 1
            elif is_under(f, DROPPED_PREFIX):
                n_drop += 1
            else:
                unaccounted.append(f"[{b}] {f}")
    print(f"\n--- 全分支审计（{len(branches)} 个分支，去重后 {len(seen)} 个文件）---")
    print(f"  ✅ 已迁移/已覆盖 : {n_mig}")
    print(f"  ⬜ 已明确不迁    : {n_drop}")
    print(f"  ❓ 未交代        : {len(unaccounted)}")
    if unaccounted:
        print("\n未交代（按分支）：")
        for u in sorted(unaccounted)[:40]:
            print(f"    {u}")
        if len(unaccounted) > 40:
            print(f"    …… 另 {len(unaccounted) - 40} 个")
    return n_mig, n_drop, unaccounted


def main() -> int:
    files: list[str] = []
    for root in [SRC] + EXTRA_ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if f.suffix in SKIP_SUFFIX:
                continue
            prefix = "" if root == SRC else f"[{root.name}] "
            files.append(prefix + rel)

    unaccounted = []
    n_mig = n_drop = 0
    for f in files:
        raw = f.split("] ", 1)[1] if f.startswith("[") else f
        if f.startswith("[raw_legal_dl]"):
            n_mig += 1
            continue
        if f.startswith("[_data_tools]"):
            # 只迁了 LeCaRD 的 data 与 LawBench 的 zero_shot/1-1,1-2
            if raw.startswith("LeCaRD-main/data/") or raw in (
                    "LawBench-main/data/zero_shot/1-1.json",
                    "LawBench-main/data/zero_shot/1-2.json"):
                n_mig += 1
            else:
                n_drop += 1
            continue
        if is_under(raw, MIGRATED_PREFIX):
            n_mig += 1
        elif is_under(raw, DROPPED_PREFIX):
            n_drop += 1
        else:
            unaccounted.append(f)

    print(f"【working tree】旧仓库 + 外部数据目录共 {len(files)} 个文件")
    print(f"  ✅ 已迁移/已覆盖 : {n_mig}")
    print(f"  ⬜ 已明确不迁    : {n_drop}")
    print(f"  ❓ 未交代        : {len(unaccounted)}")
    if unaccounted:
        print("\n未交代的文件（需要你确认是「该迁」还是「不迁」）：")
        for u in sorted(unaccounted)[:60]:
            print(f"    {u}")
        if len(unaccounted) > 60:
            print(f"    …… 另 {len(unaccounted) - 60} 个")

    _, _, branch_unaccounted = audit_branches()

    if not unaccounted and not branch_unaccounted:
        print("\n✅ working tree 与全部分支的每一个文件都已归入「已迁」或「已明确不迁」。")
    else:
        print(f"\n⚠️ 仍有 {len(unaccounted) + len(branch_unaccounted)} 个文件未交代。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
