# -*- coding: utf-8 -*-
"""检索链路探针：对单条 query 打印每个阶段的候选（file / status_code / score）。

用途：定位"未生效版本没被捞出来"这类检索侧失败——是召回没进候选，还是 rerank 压下去，
还是时效重排没生效。不跑 LLM，纯检索，秒级。

用法：
    python scripts/probe_retrieval.py "《中华人民共和国农业法》现在生效了吗？什么时候施行？"
    python scripts/probe_retrieval.py "<query>" --top-k 5 --currency on
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QUERY_REWRITE_MODE", "off")


def _view(sn) -> dict:
    meta = dict(getattr(getattr(sn, "node", None), "metadata", None) or {})
    return {
        "file": Path(str(meta.get("file_path") or meta.get("file_name") or "")).name,
        "status_code": meta.get("status_code"),
        "effective_date": meta.get("effective_date"),
        "score": round(float(getattr(sn, "score", 0.0) or 0.0), 4),
    }


def _print_stage(name: str, rows: list[dict], *, count: int | None = None, limit: int = 20) -> None:
    print(f"\n── {name}（count={count if count is not None else len(rows)}，"
          f"trace 只留前 {len(rows)} 条）")
    for r in rows[:limit]:
        print(f"   sc={r.get('status_code')}  score={r.get('score'):<8} {r.get('file')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--currency", choices=["on", "off"], default="on")
    args = ap.parse_args()

    from src.config_LEGACY_REFERENCE import get_settings
    from src.currency import apply_currency_rerank, detect_status_intent, extract_year_anchor
    from src.retrieval import retrieve_scored_nodes
    from src.trace import create_trace
    from src.vector_store import get_vector_index

    base = get_settings()
    s = base.model_copy(update={
        "cache_enabled": False,
        "query_rewrite_mode": "off",
        "retrieval_early_gate_enabled": True,
        "currency_rerank_enabled": args.currency == "on",
    })

    q = args.query
    print(f"query: {q}")
    print(f"top_k={args.top_k} currency={args.currency} "
          f"status_intent={detect_status_intent(q)} year_anchor={extract_year_anchor(q)}")
    print(f"candidate_top_k(rerank)={s.rerank_candidate_top_k} "
          f"bm25_candidate_top_k={s.bm25_candidate_top_k} rerank_enabled={s.rerank_enabled}")

    index = get_vector_index()
    tr = create_trace()
    sr = retrieve_scored_nodes(index=index, user_query=q, top_k=args.top_k, settings=s, trace=tr)

    from src.trace import load_trace
    for st in load_trace(tr.query_id):
        stage = st.get("stage")
        if stage in ("vector", "bm25", "fuse", "rerank"):
            _print_stage(stage, st.get("top") or [], count=st.get("count"))
        elif stage == "currency":
            print(f"\n── currency: {st.get('mode')} "
                  f"pending_boosted={st.get('pending_boosted')} "
                  f"in={st.get('in_count')} out={st.get('out_count')}")
        elif stage in ("vector_top1", "route", "parallel"):
            print(f"\n── {stage}: {json.dumps({k: v for k, v in st.items() if k not in ('ts', 'query_id', 'stage', 'elapsed_ms')}, ensure_ascii=False)}")

    print("\n══ 最终返回（top_k）")
    for r in [_view(sn) for sn in sr.nodes]:
        print(f"   sc={r['status_code']}  eff={r['effective_date']}  score={r['score']:<8} {r['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())