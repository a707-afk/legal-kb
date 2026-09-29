# -*- coding: utf-8 -*-
"""从旧仓库的 **main 分支** 提取文件到本仓库。

为什么需要这个脚本
------------------
前四轮迁移只在 `feature/ecom-agent` 的 working tree 上做，**从未检查其他分支**。
而 main 分支相对 feature 有 **141 个独有文件**，其中包含几个"以为要重写、其实已有实现"的模块：

    app/agent_grader.py         Grader（D5 的确定性判据）
    app/domain_router.py        分库路由（D8-1 要的，我之前判断"已被删除需重建"）
    app/llm_zhipu.py            智谱客户端（我之前计划"D2 改为智谱"）
    app/services/session_mgr.py 会话/记忆管理（我列为 P2 缺口）
    app/agent/tools.py          完整工具集（迁过来的 research_tools 只有 2 个工具）
    app/structured_logging.py   结构化日志
    app/behavior_guard.py       行为护栏（且带评测）
    data/domain_router_profiles.json   路由 profile 数据

**教训：迁移审计必须覆盖"所有分支 + working tree + git 历史"，不只是当前分支。**

用法
----
    python scripts/migrate_from_main_branch.py --dry-run
    python scripts/migrate_from_main_branch.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

# 旧仓库路径：迁移时的一次性输入，用环境变量 LEGACY_REPO 指定（避免把本机绝对路径写进仓库）
SRC = Path(os.environ.get("LEGACY_REPO", r"../rag-kb-project")).resolve()
DST = Path(__file__).resolve().parent.parent
BRANCH = "main"

# (main 分支路径, 新项目路径, 归类, 说明)
ITEMS = [
    # ---------- P0：直接影响 D5 / D8 的实现 ----------
    ("app/agent_grader.py", "src/agent_grader.py", "code",
     "★ Grader——D5 的「证据充分性判定」组件，此前判断为「要补」"),
    ("app/domain_router.py", "src/domain_router.py", "code",
     "★ 分库路由——D8-1 对照实验需要；此前判断为「已被删除，需重建」"),
    ("app/llm_zhipu.py", "src/llm_zhipu.py", "code",
     "★ 智谱客户端——一代就实现了；此前计划「D2 改为智谱」"),
    ("app/services/session_mgr.py", "src/session_mgr.py", "code",
     "★ 会话/记忆管理——此前列为 P2 缺口「无跨会话记忆」"),
    ("app/agent/tools.py", "src/agent/tools.py", "code",
     "★ 完整工具集（14.9KB）——迁过来的 research_tools 只有 2 个工具"),
    ("app/structured_logging.py", "src/structured_logging.py", "code",
     "结构化日志（与 src/logging_utils.py 互补）"),
    ("app/behavior_guard.py", "src/behavior_guard.py", "code",
     "行为护栏，且带完整评测（BEHAVIOR-GUARD-EVAL）"),
    ("app/retrieval_intent_boost.py", "src/retrieval_intent_boost.py", "code",
     "检索意图加权（retrieval.py 里注明「已移除」，实现仍在 main）"),
        ("app/vector_backend.py", "src/vector_backend.py", "code", "向量后端抽象"),
    ("tests/test_agent_grader.py", "tests/test_agent_grader.py", "code",
     "★ Grader 的测试"),

    # ---------- P0：评测数据（可复现对照实验的原料） ----------
    ("data/domain_router_profiles.json", "data/router/domain_router_profiles.json", "data",
     "★ 路由 profile 数据——重建路由必需"),
    ("data/domain_router_profiles_cs.json", "data/router/domain_router_profiles_cs.json", "data", "路由 profile（客服版）"),
    ("data/router_calibration.default.json", "data/router/router_calibration.default.json", "data", "路由校准参数"),
    ("data/router_eval_golden.jsonl", "eval/legacy-v3/router_eval_golden.jsonl", "data",
     "★ 路由评测金标（80 条）"),
    ("data/eval_access_control_questions.jsonl", "eval/legacy-v3/eval_access_control_questions.jsonl", "data",
     "★ 权限实验的原始题集（42%→88.5% 那个实验）"),
    ("data/eval_hallucination.jsonl", "eval/legacy-v3/eval_hallucination.jsonl", "data",
     "幻觉评测题（8 条，零区分度那批）"),

    # ---------- P1：旧版策略文档（旧日志说 main 有「16 条决策账本 + 旧版策略文档」） ----------
    ("docs/LEGACY-NOTES.md", "docs/legacy-main/LEGACY-NOTES.md", "doc", "★ 旧版策略文档"),
    ("docs/DECISION-LOG.md", "docs/legacy-main/DECISION-LOG.md", "doc",
     "★ 决策账本（main 版本，可能与 feature 的 docs/archive 版不同）"),
    ("docs/PROJECT-1-AGENTIC-RAG.md", "docs/legacy-main/PROJECT-1-AGENTIC-RAG.md", "doc", "项目一剧本"),
    ("docs/PROJECT-1-PLAYBOOK.md", "docs/legacy-main/PROJECT-1-PLAYBOOK.md", "doc", "项目一手册"),
    ("docs/PROJECT-2-MULTI-AGENT.md", "docs/legacy-main/PROJECT-2-MULTI-AGENT.md", "doc", "项目二剧本"),
    ("docs/PROJECT-2-PLAYBOOK.md", "docs/legacy-main/PROJECT-2-PLAYBOOK.md", "doc", "项目二手册"),
    ("docs/STRATEGY-GUIDE.md", "docs/legacy-main/STRATEGY-GUIDE.md", "doc", "策略指南"),
    ("docs/RAG-AGENT-KNOWLEDGE-MAP.md", "docs/legacy-main/RAG-AGENT-KNOWLEDGE-MAP.md", "doc", "知识地图"),
    ("docs/CURRENT-PROJECT-DEEP-PLAN.md", "docs/legacy-main/CURRENT-PROJECT-DEEP-PLAN.md", "doc", "深度计划"),
    ("TECH-V1.md", "docs/legacy-main/TECH-V1.md", "doc", "技术方案 v1"),
    ("REQUIREMENTS-ONEPAGER.md", "docs/legacy-main/REQUIREMENTS-ONEPAGER.md", "doc", "需求一页纸"),
    ("BUILD_REPORT.md", "docs/legacy-main/BUILD_REPORT.root.md", "doc", "构建报告（根目录版）"),
    ("旧版说明.md", "docs/legacy-main/旧版说明.md", "doc", "旧版说明文档"),
    (".planning/ROADMAP.md", "docs/legacy-main/ROADMAP.md", "doc", "路线图"),
    (".planning/STATE.md", "docs/legacy-main/STATE.md", "doc", "项目状态"),

    # ---------- P1：实验文档与结果（对照基线的原始材料） ----------
    ("docs/ACCESS-CONTROL-EVAL.md", "eval/legacy-v3/ACCESS-CONTROL-EVAL.md", "doc", "权限评测报告"),
    ("docs/ACCESS-CONTROL-EVAL-CHROMA.md", "eval/legacy-v3/ACCESS-CONTROL-EVAL-CHROMA.md", "doc", "权限评测（Chroma）"),
    ("docs/ACCESS-CONTROL-EVAL-QDRANT.md", "eval/legacy-v3/ACCESS-CONTROL-EVAL-QDRANT.md", "doc", "权限评测（Qdrant）"),
    ("docs/ROUTER-EVAL.md", "eval/legacy-v3/ROUTER-EVAL.md", "doc", "路由评测方法"),
    ("docs/BEHAVIOR-GUARD-EVAL.md", "eval/legacy-v3/BEHAVIOR-GUARD-EVAL.md", "doc", "行为护栏评测"),
    ("docs/G-K-METRICS-SUMMARY.md", "eval/legacy-v3/G-K-METRICS-SUMMARY.md", "doc", "指标汇总"),
    ("docs/SCALE-BENCHMARKS.md", "eval/legacy-v3/SCALE-BENCHMARKS.md", "doc", "规模压测（5070 那台机器的数据）"),
    ("docs/EVAL-BASELINE-COMPARISON.md", "eval/legacy-v3/EVAL-BASELINE-COMPARISON.md", "doc", "基线对照"),
    ("docs/router_eval_metrics_summary.json", "eval/legacy-v3/router_eval_metrics_summary.json", "data", "路由评测结果"),
    ("docs/router_eval_metrics_aggregate.csv", "eval/legacy-v3/router_eval_metrics_aggregate.csv", "data", "路由评测结果"),
    ("docs/router_eval_metrics_confusion.csv", "eval/legacy-v3/router_eval_metrics_confusion.csv", "data", "路由混淆矩阵"),
    ("docs/eval_access_control_qdrant.json", "eval/legacy-v3/eval_access_control_qdrant.json", "data", "权限评测原始结果"),
    ("docs/eval_qdrant_vs_chroma_access.json", "eval/legacy-v3/eval_qdrant_vs_chroma_access.json", "data", "后端对比"),
    ("docs/eval_behavior_guard.json", "eval/legacy-v3/eval_behavior_guard.json", "data", "护栏评测原始结果"),
    ("docs/eval_hallucination.json", "eval/legacy-v3/eval_hallucination.json", "data", "幻觉评测原始结果"),
    ("docs/eval_four_baselines_summary.json", "eval/legacy-v3/eval_four_baselines_summary.json", "data", "四组基线"),
    ("docs/eval_prod_router_matrix_summary.json", "eval/legacy-v3/eval_prod_router_matrix_summary.json", "data", "路由矩阵"),

    # ---------- P2：可复用的评测脚本 ----------
    ("scripts/run_eval_router.py", "eval/legacy-v3/scripts/run_eval_router.py", "code", "路由评测 runner"),
    ("scripts/run_eval_access_control.py", "eval/legacy-v3/scripts/run_eval_access_control.py", "code", "权限评测 runner"),
    ("scripts/run_eval_behavior_guard.py", "eval/legacy-v3/scripts/run_eval_behavior_guard.py", "code", "护栏评测 runner"),
    ("scripts/run_eval_hallucination.py", "eval/legacy-v3/scripts/run_eval_hallucination.py", "code", "幻觉评测 runner"),
    ("scripts/generate_router_eval_golden.py", "eval/legacy-v3/scripts/generate_router_eval_golden.py", "code", "路由金标生成"),
    ("scripts/run_eval_hybrid_rrf_production.py", "eval/legacy-v3/scripts/run_eval_hybrid_rrf_production.py", "code",
     "★ RRF 生产评测（D-04 融合实验的现成脚本）"),
    ("scripts/run_eval_four_baselines.py", "eval/legacy-v3/scripts/run_eval_four_baselines.py", "code",
     "★ 四组基线 runner（D8-1 对照实验的现成编排）"),
    ("scripts/compare_access_eval_backends.py", "eval/legacy-v3/scripts/compare_access_eval_backends.py", "code", "后端对比"),
]

# 明确不迁（main 分支上但与本项目无关）
SKIP = [
    ("app/agent_graph/**", "旧 LangGraph 多图实现，蓝图明确不做多 Agent 实体"),
    ("app/api/chat.py", "旧 chat 路由，feature 分支的 routes_rag.py 更新"),
    ("app/llm_zhipu.py 之外的多语言/客服脚本", "—"),
    ("scripts/download_cs_data.py / preprocess_cs_data.py / extract_csds.py / build_cn_index.py",
     "客服语料处理管线，法律项目不用"),
    ("scripts/classify_intents.py / clean_cn_pipeline.py / run_cs_eval.py / reindex_cs.py",
     "客服时代脚本"),
    ("scripts/gradio_app.py / start_dsw.sh / run_qdrant_migrate_eval.ps1", "DSW/Gradio 部署脚本"),
    ("scripts/smoke_cs_retrieval.py / smoke_policy_llm_guard.py / run_policy_phase3_answer_sample.py", "旧冒烟脚本"),
    ("scripts/test_embed.py / verify_enterprise_chunk_metadata.py", "一次性脚本"),
    ("data/docs/** / data/docs_cs/** / data/docs_cn/** / data/bm25_cn_corpus.jsonl",
     "客服语料与课程讲义"),
    ("data/eval_cs_*.jsonl / eval_enterprise_questions.jsonl / eval_questions.jsonl", "客服时代评测题"),
    ("static/app/**", "前端构建产物"),
    ("tests/server_test.py / test_agent_graph_*.py", "对应模块不迁"),
    ("docs/PHASE-*-PROGRESS.md / MILESTONE-* / NEXT-MILESTONE.md / POLICY-MILESTONE-ACCEPTANCE.md",
     "过程性进度文档，价值已被 G-K-METRICS-SUMMARY 与 DECISION-LOG 覆盖"),
    ("docs/CURRENT-STATUS.md / QDRANT-NEXT.md / qdrant_reindex.log / DATA-CORPUS-REDESIGN.md",
     "过时状态文档"),
    (".env.cs-agent / .env.dsw", "旧环境变量文件（含旧 key，不迁）"),
]


def sh(args: list[str]) -> str:
    r = subprocess.run(args, cwd=SRC, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exists = set(sh(["git", "ls-tree", "-r", "--name-only", BRANCH]).splitlines())
    print(f"{BRANCH} 分支共 {len(exists)} 个文件；本脚本要提取 {len(ITEMS)} 个\n")

    manifest, n_ok, n_missing, n_skip = [], 0, 0, 0
    for src_rel, dst_rel, kind, why in ITEMS:
        dst = DST / dst_rel
        if src_rel not in exists:
            print(f"  MISSING        {src_rel}")
            n_missing += 1
            manifest.append({"src": src_rel, "dst": dst_rel, "why": why, "status": "MISSING"})
            continue
        if dst.exists():
            print(f"  skip(exists)   {src_rel:52s} -> {dst_rel}")
            manifest.append({"src": src_rel, "dst": dst_rel, "why": why, "status": "skip(exists)"})
            continue
        if args.dry_run:
            print(f"  would extract  {src_rel:52s} -> {dst_rel}")
            manifest.append({"src": src_rel, "dst": dst_rel, "why": why, "status": "dry"})
            continue
        content = sh(["git", "show", f"{BRANCH}:{src_rel}"])
        if not content:
            print(f"  EMPTY          {src_rel}")
            n_missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8", newline="\n")
        print(f"  extracted      {src_rel:52s} -> {dst_rel}")
        n_ok += 1
        manifest.append({"src": src_rel, "dst": dst_rel, "why": why, "status": "extracted"})

    for path, why in SKIP:
        n_skip += 1
        manifest.append({"path": path, "why": why, "status": "dropped"})

    if not args.dry_run:
        out = DST / "reports" / "migration-manifest-main.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n清单已写入: {out}")

    print(f"\n汇总：提取 {n_ok} ｜ 已存在 {len(ITEMS) - n_ok - n_missing} ｜ 缺失 {n_missing} ｜ 明确不迁 {n_skip} 类")
    if n_missing:
        print("⚠️ 有缺失项，需人工确认路径")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
