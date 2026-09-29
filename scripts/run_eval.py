# -*- coding: utf-8 -*-
"""检索层评测 runner（❗不是系统能力评测）：读 v1.jsonl → 只跑 retrieve_scored_nodes + currency → 按 judge_type 判分。

⚠️ 定位（P1）：本脚本量的是“检索 + 确定性时效重排”的质量，**不跑 Agent、不跑 LLM 生成**。
   系统/Agent 端到端能力见 scripts/run_agent_eval.py。这里的分数不要当作“Agent 能力分”。
   overall_accuracy 含 A 类（强同源）；**能力分看 capability_accuracy_excl_A**（A 只作回归基线，P4）。

可观测性（关机不丢、可续跑）：logs/run_eval.log + reports/eval_progress.json + reports/eval_rows.jsonl。

可观测性（这次重写补上——上一版只在最后写报告，中途看不到进度、关机全丢）
--------------------------------------------------------------------------
- **实时日志**：`logs/run_eval.log`（逐行 flush，不依赖终端缓冲）
- **进度文件**：`reports/eval_progress.json`（done/total/percent/rate/eta/current_id，每题更新）
- **增量落盘**：`reports/eval_rows.jsonl`（每判一题追加一行）→ 关机也保留已完成部分
- **续跑**：`--resume` 跳过 eval_rows.jsonl 里已判过的 (id, config)

速度：baseline 与 currency 的检索**完全相同**（currency 只是精排后的重排），
所以 `--both` **只检索一次、判两遍**（currency 结果 = 对同一批候选调 apply_currency_rerank），
比跑两遍快一倍，且 A/B 更干净（唯一变量就是 currency）。

判分全部基于**检索结果**（不跑 LLM 生成），可复现、快、零 API 成本；
F 类 should_disclaim 需生成，标 generation_required 跳过。

用法
----
    python scripts/run_eval.py --both                    # 基线 + 时效重排对照（推荐）
    python scripts/run_eval.py --both --resume           # 关机/中断后续跑
    python scripts/run_eval.py --config baseline --limit 5   # 小样冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SET_PATH = ROOT / "eval" / "sets" / "v1.jsonl"
REPORT_DIR = ROOT / "reports"
LOG_PATH = ROOT / "logs" / "run_eval.log"
PROGRESS_PATH = REPORT_DIR / "eval_progress.json"
ROWS_PATH = REPORT_DIR / "eval_rows.jsonl"

# 过期法条率只对“应返回现行有效版”的题统计；这些题的正确 top-1 本就不是 status_code=3，须排除：
#   version_time_travel（要历史版）/ version_enumerate（要列全部版本）/ not_yet_effective（问的就是未生效版）
#   / out_of_corpus（应拒答）/ status_unlabeled（无时效元数据）
STALE_EXCLUDE_DIFFICULTY = {
    "version_time_travel", "version_enumerate", "not_yet_effective",
    "out_of_corpus", "status_unlabeled",
}


# ── 可观测性：tee 日志 + 进度文件 ──────────────────────────────

class _Tee:
    """同时写终端和日志文件；日志侧每行 flush（终端缓冲不可靠）。"""

    def __init__(self, stream, fh) -> None:
        self._stream, self._fh = stream, fh

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass
        return len(data)

    def flush(self):
        for h in (self._stream, self._fh):
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self):  # pragma: no cover
        try:
            return bool(self._stream.isatty())
        except Exception:  # noqa: BLE001
            return False


def _attach_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = LOG_PATH.open("a", encoding="utf-8", errors="replace")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
        setattr(sys, name, _Tee(stream, fh))
    return fh


def write_progress(**kw):
    kw["updated_at"] = datetime.now().isoformat(timespec="seconds")
    kw["pid"] = os.getpid()
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_PATH.write_text(json.dumps(kw, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# ── 判分 ────────────────────────────────────────────────────────

def _bn(p) -> str:
    s = str(p or "").replace("\\", "/").rsplit("/", 1)[-1]
    return s[:-3] if s.endswith(".md") else s


def _node_view(sn) -> dict:
    m = getattr(sn.node, "metadata", None) or {}
    fp = m.get("file_path") or m.get("file_name") or ""
    sc = m.get("status_code")
    try:
        sc = int(sc) if sc is not None else None
    except (TypeError, ValueError):
        sc = None
    return {
        "file": fp, "bn": _bn(fp), "status_code": sc,
        "title": m.get("title") or "", "effective_date": str(m.get("effective_date") or ""),
        "heading": m.get("heading_path") or m.get("header_path") or m.get("heading") or "",
        # 判分窗口 2000 字符（原为 300）：A 类 file_and_article_hit 要在正文里找条号，
        # 而「第十八条」这类条号常出现在 300 字符之后（前置标题/章名/编章序言占位），
        # 300 会把命中判成未命中（假阴性）。该字段只用于判分、不落盘，放宽无成本。
        "text": (sn.node.get_content() or "")[:2000] if hasattr(sn.node, "get_content") else "",
        "score": float(sn.score or 0.0),
    }


def judge(item: dict, retrieved: list[dict], gate_passed: bool):
    jt = item["judge"]["type"]
    gt = item.get("gt", {})
    bns = [r["bn"] for r in retrieved]

    if jt == "file_and_article_hit":
        exp = _bn(gt.get("expected_file"))
        art = gt.get("article", "")
        for r in retrieved:
            if r["bn"] == exp and (art in r["heading"] or art in r["text"]):
                return True, {"matched": r["file"]}
        return False, {"expected": exp, "got_top": bns[:3]}
    if jt == "expected_source_recall":
        return _bn(gt.get("expected_file")) in bns, {}
    if jt == "multi_source_recall":
        exp = {_bn(f) for f in gt.get("expected_files", [])}
        hit = len(exp & set(bns))
        return hit >= gt.get("min_sources", 2), {"hit": hit, "need": gt.get("min_sources", 2)}
    if jt == "top1_status_code_eq":
        want = item["judge"].get("value", gt.get("expected_status_code", 3))
        sc = retrieved[0]["status_code"] if retrieved else None
        return sc == want, {"got_status": sc, "want_status": want}
    if jt == "version_set_recall":
        vs = gt.get("versions", [])
        got = sum(1 for v in vs if v.get("bbbs") and any(v["bbbs"] in b for b in bns))
        return got >= 2, {"versions_total": len(vs), "retrieved": got}
    if jt == "version_effective_at_year":
        exp = _bn(gt.get("expected_file"))
        rank = bns.index(exp) if exp in bns else None
        return (rank == 0), {"expected": exp, "rank": rank, "got_top1": bns[0] if bns else None}
    if jt == "detect_not_effective":
        exp = _bn(gt.get("expected_file"))
        rank = bns.index(exp) if exp in bns else None
        return (rank is not None and rank < 3), {"expected": exp, "rank": rank}
    if jt == "hierarchy_higher_wins":
        hi = _bn(gt.get("higher_rank_file"))
        return (hi in bns, {"higher_in_topk": hi in bns, "higher_rank": bns.index(hi) if hi in bns else None})
    if jt == "should_refuse":
        return (not gate_passed), {"gate_passed": gate_passed}
    if jt == "should_disclaim_unknown_effectiveness":
        return None, {"reason": "generation_required"}
    return None, {"reason": f"unknown_judge:{jt}"}


def _gate(nodes, settings) -> bool:
    from src.gates import evaluate_similarity_gate
    try:
        return bool(evaluate_similarity_gate(nodes, settings).passed)
    except Exception:  # noqa: BLE001
        return bool(nodes)


# ── 统计 ────────────────────────────────────────────────────────

def _summarize(rows: list[dict], config: str) -> dict:
    per_cat = defaultdict(lambda: {"n": 0, "pass": 0, "skip": 0})
    stale = stale_denom = stale_unknown = 0
    for r in rows:
        if r["config"] != config:
            continue
        d = per_cat[r["cat"]]
        d["n"] += 1
        if r["passed"] is None:
            d["skip"] += 1
        elif r["passed"]:
            d["pass"] += 1
        # 分母 = **在范围内的全部题**（只看 difficulty），不再要求 top1_status 非空。
        # 旧口径按「top1_status 非空」计数，而 baseline/currency 的 top-1 不同 →
        # 分母会差 1（149 vs 150），两个 config 的过期率不可直接比。
        if r.get("difficulty") not in STALE_EXCLUDE_DIFFICULTY:
            stale_denom += 1
            if r.get("top1_status") is None:
                stale_unknown += 1
            elif r["top1_status"] != 3:
                stale += 1
    summary = {}
    tn = tp = ts = 0
    tn_x = tp_x = ts_x = 0
    for c in "ABCDEF":
        d = per_cat.get(c, {"n": 0, "pass": 0, "skip": 0})
        judged = d["n"] - d["skip"]
        summary[c] = {"n": d["n"], "judged": judged, "pass": d["pass"], "skip": d["skip"],
                      "accuracy": round(d["pass"] / judged, 4) if judged else None}
        tn += d["n"]
        tp += d["pass"]
        ts += d["skip"]
        if c != "A":  # A 强同源、只作回归基线，不计入“检索层能力分”（P4）
            tn_x += d["n"]
            tp_x += d["pass"]
            ts_x += d["skip"]
    return {
        "config": config, "n_items": tn, "per_category": summary,
        "overall_accuracy": round(tp / (tn - ts), 4) if (tn - ts) else None,
        "capability_accuracy_excl_A": round(tp_x / (tn_x - ts_x), 4) if (tn_x - ts_x) else None,
        "stale_top1_rate": round(stale / stale_denom, 4) if stale_denom else None,
        "stale_top1": stale, "stale_denom": stale_denom,
        # top1_status 为空的题（检索无结果）：不计入 stale 分子，但单独报出来，
        # 避免「无结果」被悄悄当成「不过期」而美化过期率
        "stale_unknown_status": stale_unknown,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="D4 评测 runner（可观测 + 可续跑）")
    ap.add_argument("--config", choices=["baseline", "currency"], default=None)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="跳过 eval_rows.jsonl 里已判过的题")
    args = ap.parse_args()

    log_fh = _attach_log()
    if not SET_PATH.is_file():
        print(f"❌ 评测集不存在：{SET_PATH}", file=sys.stderr)
        return 1
    items = [json.loads(line) for line in SET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    want = ["baseline", "currency"] if args.both else [args.config or "baseline"]
    print(f"载入 {len(items)} 题；configs={want}；top_k={args.top_k}；resume={args.resume}", flush=True)

    # 增量结果 + 续跑
    existing: list[dict] = []
    if args.resume and ROWS_PATH.is_file():
        existing = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    done_pairs = {(r["id"], r["config"]) for r in existing}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows_fh = ROWS_PATH.open("a" if args.resume else "w", encoding="utf-8")
    all_rows = list(existing)

    from src.config_LEGACY_REFERENCE import get_settings
    from src.currency import apply_currency_rerank
    from src.retrieval import retrieve_scored_nodes
    from src.trace import create_trace, load_trace
    from src.vector_store import get_vector_index

    base = get_settings()
    # 评测锁定：关缓存（每条真跑）+ 关 query_rewrite（auto 模式会对带“？/什么”的题调 LLM 改写，
    # 撞限流+网络延迟→每题卡几分钟；检索评测应隔离改写，且要可复现）
    _common = {"cache_enabled": False, "query_rewrite_mode": "off", "retrieval_early_gate_enabled": True}
    s_base = base.model_copy(update={**_common, "currency_rerank_enabled": False})
    s_cur = base.model_copy(update={**_common, "currency_rerank_enabled": True})
    index = get_vector_index()
    print("索引已加载，开始评测…", flush=True)

    todo = [it for it in items if not all((it["id"], c) in done_pairs for c in want)]
    t0 = time.perf_counter()
    for i, item in enumerate(todo):
        q = item["question"]
        row_base = {"id": item["id"], "cat": item["category"], "config": "baseline", "difficulty": item.get("difficulty")}
        row_cur = {"id": item["id"], "cat": item["category"], "config": "currency", "difficulty": item.get("difficulty")}
        try:
            tr = create_trace(query_id=f"eval-{item['id']}")
            t_item = time.perf_counter()
            sr = retrieve_scored_nodes(index=index, user_query=q, top_k=args.top_k, settings=s_base, trace=tr)
            lat = round((time.perf_counter() - t_item) * 1000, 1)
            bvs = eg = None
            for st in load_trace(tr.query_id):
                if st.get("stage") == "vector_top1":
                    bvs, eg = st.get("best_vector_score"), st.get("early_gate")
            if "baseline" in want:
                bv = [_node_view(sn) for sn in sr.nodes]
                p, info = judge(item, bv, _gate(sr.nodes, s_base))
                row_base.update(passed=p, top1=bv[0]["bn"] if bv else None,
                                top1_status=bv[0]["status_code"] if bv else None, n_hits=len(bv),
                                latency_ms=lat, best_vector_score=bvs, early_gate=eg, **info)
            if "currency" in want:
                cn = apply_currency_rerank(sr.nodes, q, s_cur)
                cv = [_node_view(sn) for sn in cn]
                p, info = judge(item, cv, _gate(cn, s_cur))
                row_cur.update(passed=p, top1=cv[0]["bn"] if cv else None,
                               top1_status=cv[0]["status_code"] if cv else None, n_hits=len(cv),
                               latency_ms=lat, **info)
        except Exception as e:  # noqa: BLE001  单题失败不拖垮整轮
            for rr in (row_base, row_cur):
                rr.setdefault("passed", None)
                rr["error"] = f"{type(e).__name__}: {str(e)[:120]}"

        for c, rr in (("baseline", row_base), ("currency", row_cur)):
            if c in want:
                rows_fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
                all_rows.append(rr)
        rows_fh.flush()

        done = i + 1
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done) / rate if rate > 0 else 0
        write_progress(phase="eval", configs=want, done=done, total=len(todo),
                       percent=round(100 * done / len(todo), 1) if todo else 100,
                       elapsed_s=round(elapsed, 1), rate_per_s=round(rate, 3),
                       eta_s=round(eta, 1), current_id=item["id"])
        print(f"[{done}/{len(todo)}] {item['id']} base={row_base.get('passed')} "
              f"cur={row_cur.get('passed')} | {elapsed:.0f}s eta {eta:.0f}s", flush=True)

    rows_fh.close()

    # 汇总
    results = {}
    for c in want:
        results[c] = _summarize(all_rows, c)
        s = results[c]
        print(f"\n=== {c} ===  能力分(不含A)={s['capability_accuracy_excl_A']}  overall(含A)={s['overall_accuracy']}  "
              f"过期法条率={s['stale_top1_rate']} ({s['stale_top1']}/{s['stale_denom']}, 无状态 {s['stale_unknown_status']})", flush=True)
        for cat in "ABCDEF":
            d = s["per_category"][cat]
            print(f"   {cat}: acc={d['accuracy']} ({d['pass']}/{d['judged']}, skip {d['skip']})", flush=True)

    if "baseline" in results and "currency" in results:
        b, cu = results["baseline"], results["currency"]
        cmp_rows = []
        for cat in "ABCDEF":
            ba = b["per_category"][cat]["accuracy"]
            ca = cu["per_category"][cat]["accuracy"]
            cmp_rows.append({"category": cat, "baseline": ba, "currency": ca,
                             "delta": round(ca - ba, 4) if (ba is not None and ca is not None) else None})
        results["_comparison"] = {
            "per_category": cmp_rows,
            "capability_excl_A": {"baseline": b["capability_accuracy_excl_A"], "currency": cu["capability_accuracy_excl_A"],
                                  "delta": round((cu["capability_accuracy_excl_A"] or 0) - (b["capability_accuracy_excl_A"] or 0), 4)},
            "overall_incl_A": {"baseline": b["overall_accuracy"], "currency": cu["overall_accuracy"],
                               "delta": round((cu["overall_accuracy"] or 0) - (b["overall_accuracy"] or 0), 4)},
            "stale_top1_rate": {"baseline": b["stale_top1_rate"], "currency": cu["stale_top1_rate"]},
        }
        print("\n=== 时效重排对照（currency − baseline）===", flush=True)
        for row in cmp_rows:
            print(f"   {row['category']}: {row['baseline']} → {row['currency']}  (Δ {row['delta']})", flush=True)
        print(f"   过期法条率: {b['stale_top1_rate']} → {cu['stale_top1_rate']}", flush=True)

    tag = "both" if args.both else (args.config or "baseline")
    out = REPORT_DIR / f"eval_{tag}_report.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_progress(phase="done", configs=want, done=len(todo), total=len(todo), percent=100,
                   elapsed_s=round(time.perf_counter() - t0, 1), report=str(out.relative_to(ROOT)))
    print(f"\n报告: {out.relative_to(ROOT)}｜逐题明细: {ROWS_PATH.relative_to(ROOT)}", flush=True)
    try:
        from src.vector_store import clear_index_memory_cache
        clear_index_memory_cache()
    except Exception:  # noqa: BLE001
        pass
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
