# -*- coding: utf-8 -*-
"""检索冒烟测试：端到端验证索引可用 + 产出真实 trace（D2 验收）。

验收口径（PLAN.md D2）
--------------------
- 集合点数 ≥ 3 万
- 重复节点自检（同位置）= 0
- **冷启动检索 < 15s，热检索 < 1s**
- `replay_trace.py <query_id>` 能打印完整链路

用法
----
    python scripts/smoke_retrieval.py                    # 默认 5 条内置查询
    python scripts/smoke_retrieval.py --query "试用期工资最低能给多少"
    python scripts/smoke_retrieval.py --top-k 5 --repeat 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports" / "smoke_retrieval_report.json"

DEFAULT_QUERIES = [
    # 口语化（B 类形态）
    "公司拖欠工资，员工可以解除劳动合同吗？",
    # 法条精确（A 类形态）
    "《中华人民共和国劳动合同法》第三十八条的内容是什么？",
    # 跨法条（C 类形态）
    "合同没约定违约金，一方违约造成损失怎么赔？能调整吗？",
    # 版本/时效（D 类形态）
    "城市房地产管理法关于土地使用权出让是怎么规定的？",
    # 拒答形态（F 类）
    "帮我起草一份股权代持协议",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="检索冒烟测试")
    ap.add_argument("--query", action="append", help="自定义查询（可多次）")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1, help="每条查询重复次数（测热检索）")
    args = ap.parse_args()

    from src.config_LEGACY_REFERENCE import get_settings
    from src.retrieval import retrieve_scored_nodes
    from src.trace import create_trace
    from src.vector_store import get_vector_index

    settings = get_settings()
    queries = args.query or DEFAULT_QUERIES

    t0 = time.perf_counter()
    index = get_vector_index()
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"索引加载（冷）: {cold_ms:.0f}ms ｜ 集合 {settings.qdrant_collection_name}")
    print()

    results = []
    for q in queries:
        for rep in range(args.repeat):
            tr = create_trace()
            t1 = time.perf_counter()
            out = retrieve_scored_nodes(
                index=index, user_query=q, top_k=args.top_k,
                settings=settings, trace=tr,
            )
            ms = (time.perf_counter() - t1) * 1000
            nodes = out.nodes
            with_status = [n for n in nodes if (n.node.metadata or {}).get("status_code") is not None]
            tr.finish(final_count=len(nodes), latency_ms=round(ms, 1))
            print(f"[{q[:40]}]  {ms:7.0f}ms  hits={len(nodes)}  "
                  f"query_id={tr.query_id}")
            for i, sn in enumerate(nodes[: args.top_k], 1):
                m = dict(sn.node.metadata or {})
                sc = m.get("status_code")
                print(f"    {i}. {float(sn.score or 0):.4f}  status={str(sc):<5} "
                      f"{str(m.get('file_name',''))[:34]:34s} | {str(m.get('heading_path',''))[:46]}")
            if not nodes:
                print("    （无结果）")
            results.append({
                "query": q, "repeat": rep, "query_id": tr.query_id,
                "latency_ms": round(ms, 1), "hits": len(nodes),
                "hits_with_status_code": len(with_status),
                "top1_status_code": (dict(nodes[0].node.metadata or {}).get("status_code") if nodes else None),
            })
            print()

    lat = [r["latency_ms"] for r in results]
    hot = lat[1:] if len(lat) > 1 else lat
    summary = {
        "cold_index_load_ms": round(cold_ms, 1),
        "queries": len(results),
        "latency_ms_all": lat,
        "latency_ms_p50_excluding_first": round(sorted(hot)[len(hot) // 2], 1) if hot else None,
        "acceptance": {
            "cold_load_under_15s": cold_ms < 15000,
            "hot_retrieval_under_1s": (sorted(hot)[len(hot) // 2] < 1000) if hot else None,
        },
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=" * 96)
    print(json.dumps(summary["acceptance"], ensure_ascii=False, indent=1))
    print(f"冷启动索引加载: {cold_ms:.0f}ms ｜ 检索 p50(不含首次): {summary['latency_ms_p50_excluding_first']}ms")
    print(f"报告: {REPORT.relative_to(ROOT)}")
    print(f"回放示例: python scripts/replay_trace.py {results[0]['query_id']}")

    # 显式关闭 Qdrant 客户端，避免解释器退出时 __del__ 抛 "sys.meta_path is None"
    # （本地 path 模式同款问题，见 build_index.py 末尾的处理）
    try:
        from src.vector_store import clear_index_memory_cache
        clear_index_memory_cache()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
