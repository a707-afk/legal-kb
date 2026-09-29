# -*- coding: utf-8 -*-
"""按 query_id 回放检索链路 trace（D-15）。

用法
----
    python scripts/replay_trace.py --list                  # 列出最近 20 条
    python scripts/replay_trace.py q-20260928-155358-39d932  # 回放指定 query
    python scripts/replay_trace.py --last                  # 回放最近一条
    python scripts/replay_trace.py <query_id> --json       # 原始 JSON 行

为什么要这个脚本（而不是 Prometheus）
------------------------------------
Prometheus 能看延迟与错误率，但 RAG 的四种失败——**没召回 / 排序错 /
用错片段 / 引用错条号**——在指标上完全看不出来。只有链路 trace 能回答
"这一步掉在哪"。所以 D4/D6 的失败归因必须先有它。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.trace import list_recent_traces, load_trace  # noqa: E402

# 每个 stage 展示哪些字段（其余字段照原样列出）
STAGE_FIELDS = {
    "route": ["language", "collection", "candidate_k", "rerank_enabled", "fusion"],
    "parallel": ["vector", "bm25", "timeout_s", "degraded"],
    "vector": ["count"],
    "bm25": ["count"],
    "fuse": ["count", "fusion", "vector", "bm25"],
    "rerank": ["count", "reranked_from"],
    "gate": ["count", "threshold"],
    "finish": ["total_ms", "final_count"],
}


def _fmt_top(top: list[dict], limit: int = 5) -> list[str]:
    lines = []
    for i, t in enumerate(top[:limit], 1):
        sc = t.get("status_code")
        sc_txt = "None" if sc is None else str(sc)
        lines.append(
            f"        {i}. score={t.get('score'):<8} status={sc_txt:<5} "
            f"{t.get('file', '')[:44]}  | {t.get('heading_path', '')[:48]}"
        )
    return lines


def replay(query_id: str, as_json: bool = False) -> int:
    rows = load_trace(query_id)
    if not rows:
        print(f"未找到 query_id={query_id} 的 trace。用 --list 看看有哪些。", file=sys.stderr)
        return 1

    if as_json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    first = rows[0]
    print("=" * 96)
    print(f"query_id: {query_id}")
    print(f"开始时间: {first.get('ts')}")
    if first.get("retrieval_query"):
        print(f"检索词  : {first['retrieval_query']}")
    print("=" * 96)

    prev_ms = 0.0
    for r in rows:
        stage = r.get("stage", "?")
        elapsed = float(r.get("elapsed_ms") or 0.0)
        delta = elapsed - prev_ms
        prev_ms = elapsed
        fields = STAGE_FIELDS.get(stage)
        extras = []
        if fields:
            for k in fields:
                if k in r and r[k] is not None:
                    extras.append(f"{k}={r[k]}")
        else:
            for k, v in r.items():
                if k in ("ts", "query_id", "stage", "elapsed_ms", "top"):
                    continue
                if v is not None:
                    extras.append(f"{k}={v}")
        print(f"\n[{elapsed:8.1f}ms  +{delta:7.1f}ms] {stage}")
        if extras:
            print(f"      {'  '.join(extras)}")
        top = r.get("top")
        if isinstance(top, list) and top:
            for line in _fmt_top(top):
                print(line)

    print()
    print("-" * 96)
    if any(r.get("stage") == "parallel" and r.get("degraded") for r in rows):
        degraded = next(r["degraded"] for r in rows if r.get("stage") == "parallel" and r.get("degraded"))
        print(f"⚠️ 检索并行降级: {degraded}（超时或报错，已用单路结果继续）")
    finish = next((r for r in rows if r.get("stage") == "finish"), None)
    if finish:
        print(f"总耗时: {finish.get('total_ms')}ms ｜ 最终片段: {finish.get('final_count')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="按 query_id 回放检索链路 trace")
    ap.add_argument("query_id", nargs="?", help="要回放的 query_id")
    ap.add_argument("--list", action="store_true", help="列出最近的 trace")
    ap.add_argument("--last", action="store_true", help="回放最近一条")
    ap.add_argument("--json", action="store_true", help="输出原始 JSON 行")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.list:
        items = list_recent_traces(args.limit)
        if not items:
            print("暂无 trace。先跑一次检索（或 python scripts/smoke_retrieval.py）。")
            return 0
        print(f"最近 {len(items)} 条 trace：")
        for qid, p in items:
            print(f"  {qid}   ({p.stat().st_size} B)")
        return 0

    qid = args.query_id
    if args.last or not qid:
        items = list_recent_traces(1)
        if not items:
            print("暂无 trace。", file=sys.stderr)
            return 1
        qid = items[0][0]
        print(f"(使用最近一条: {qid})\n")
    return replay(qid, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
