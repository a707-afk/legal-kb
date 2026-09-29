# -*- coding: utf-8 -*-
"""L1 · Agent 端到端评测：真跑 run_agent_harness(+LLM)，判「最终答案」，量的是**系统能力**。

与 run_eval.py 的分工（P1 的修复）
----------------------------------
  run_eval.py       = **检索层**评测：retrieve_scored_nodes + currency，不跑生成，快、可复现。
                      → 它量的是"检索+确定性重排"，**不是 Agent**。报告里不再叫 overall 能力分。
  run_agent_eval.py = **系统层**评测：run_agent_harness 全链路（规划→检索→生成→评估→回环）+ LLM，
                      判 Agent 产出的**最终答案**。慢（每题数次 LLM 调用 + 限流），用 --per-category 控规模。

判分（对最终答案，确定性优先，见各 judge）
------------------------------------------
  F out_of_corpus    → 是否**拒答**（越界/无相关法条）
  F status_unlabeled → 是否**老实说"时效未标注/无法确定"**  ★这是对通用大模型的最强论据（P5）
  E not_yet_effective→ 是否识别**"尚未生效"**（并看是否提到生效年份）
  E hierarchy        → 是否指出**高位阶法优先**
  D/A/B/C            → 答案 **grounding**（句级 n-gram 溯源，对 Agent 自己检索到的 evidence）

用法
----
    python scripts/run_agent_eval.py --per-category 5 --categories D,E,F    # 小规模端到端
    python scripts/run_agent_eval.py --per-category 10                      # 全类各 10
    python scripts/run_agent_eval.py --resume                               # 断点续跑
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 系统层配置：时效重排开（生产形态）、早退 gate 开；关 query_rewrite（隔离检索侧 LLM 改写、避免限流噪声）
os.environ.setdefault("CURRENCY_RERANK_ENABLED", "true")
os.environ.setdefault("RETRIEVAL_EARLY_GATE_ENABLED", "true")
os.environ.setdefault("QUERY_REWRITE_MODE", "off")
os.environ.setdefault("AGENT_RUN_DIR", str(ROOT / "reports" / "agent_eval_runs"))
os.environ.setdefault("AUDIT_LOG_DIR", str(ROOT / "reports" / "agent_eval_runs"))

SET_PATH = ROOT / "eval" / "sets" / "v1.jsonl"
REPORT_DIR = ROOT / "reports"
LOG_PATH = ROOT / "logs" / "run_agent_eval.log"
PROGRESS_PATH = REPORT_DIR / "agent_eval_progress.json"
ROWS_PATH = REPORT_DIR / "agent_eval_rows.jsonl"

REFUSE_MARKERS = ("超出", "未检索到", "不属于", "无法回答", "知识库中没有", "没有找到",
                  "无法提供", "不在本知识库", "无相关", "抱歉", "无法协助", "非法律", "超出了")
DISCLAIM_MARKERS = ("未标注", "无法确定", "时效不明", "建议核实", "需核实", "不确定",
                    "无法确认", "未能确认", "请核实", "以官方为准", "无法判断其时效", "缺少时效")
NOT_EFFECTIVE_MARKERS = ("尚未生效", "未生效", "尚未施行", "未施行", "尚未正式施行",
                         "将于", "生效日期为", "还未生效", "尚未开始施行")
HIERARCHY_MARKERS = ("为准", "优先适用", "效力更高", "上位法", "高于", "以法律为准", "效力层级", "优先")


# ── 可观测性 ────────────────────────────────────────────────────

class _Tee:
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


# ── 判分（对最终答案）────────────────────────────────────────────

def judge_answer(item: dict, answer: str, evidence: list[dict]):
    diff = item.get("difficulty")
    gt = item.get("gt", {})
    a = (answer or "").strip()

    gr = None
    if a and evidence:
        try:
            from src.citation import sentence_level_grounding
            rep = sentence_level_grounding(a, evidence, prefer_embedding=False)
            gr = {"passed": bool(rep.passed), "unsupported_rate": round(rep.unsupported_sentence_rate, 3)}
        except Exception as e:  # noqa: BLE001
            gr = {"passed": None, "err": str(e)[:60]}
    info = {"ans_len": len(a), "evidence_n": len(evidence), "grounding": gr, "difficulty": diff}

    if not a:
        return False, {**info, "reason": "empty_answer"}
    if diff == "out_of_corpus":
        return any(m in a for m in REFUSE_MARKERS), {**info, "judge": "refuse"}
    if diff == "status_unlabeled":
        return any(m in a for m in DISCLAIM_MARKERS), {**info, "judge": "disclaim"}
    if diff == "not_yet_effective":
        hit = any(m in a for m in NOT_EFFECTIVE_MARKERS)
        yr = str(gt.get("effective_date", ""))[:4]
        return hit, {**info, "judge": "not_effective", "eff_year_mentioned": (yr in a) if yr else None}
    if diff == "hierarchy_conflict":
        hi = str(gt.get("higher_rank_title", "") or "")
        return (any(m in a for m in HIERARCHY_MARKERS) and (hi[:5] in a if hi else True)), {**info, "judge": "hierarchy"}
    # D 版本类 + A/B/C：以 grounding 为准（答案是否有据可溯）
    if gr is not None and gr.get("passed") is not None:
        return bool(gr["passed"]), {**info, "judge": "grounding"}
    return None, {**info, "judge": "no_grounding"}


def _summarize(rows: list[dict]) -> dict:
    per = defaultdict(lambda: {"n": 0, "pass": 0, "skip": 0})
    grounded = grounded_n = 0
    for r in rows:
        d = per[r.get("difficulty") or "?"]
        d["n"] += 1
        if r.get("passed") is None:
            d["skip"] += 1
        elif r.get("passed"):
            d["pass"] += 1
        g = (r.get("grounding") or {})
        if g.get("passed") is not None:
            grounded_n += 1
            grounded += 1 if g["passed"] else 0
    out = {}
    for k, d in per.items():
        j = d["n"] - d["skip"]
        out[k] = {"n": d["n"], "judged": j, "pass": d["pass"], "skip": d["skip"],
                  "accuracy": round(d["pass"] / j, 4) if j else None}
    return {"per_difficulty": out,
            "grounding_rate": round(grounded / grounded_n, 4) if grounded_n else None,
            "grounded_n": grounded_n}


async def _run_all(items, rows_fh, resume_rows):
    from src.agent.harness import run_agent_harness

    all_rows = list(resume_rows)
    t0 = time.perf_counter()
    for i, item in enumerate(items):
        write_progress(phase="agent_eval", done=i, total=len(items),
                       percent=round(100 * i / len(items), 1) if items else 100,
                       elapsed_s=round(time.perf_counter() - t0, 1),
                       eta_s=round((time.perf_counter() - t0) / max(1, i) * (len(items) - i), 1),
                       current_id=item["id"])
        row = {"id": item["id"], "cat": item["category"], "difficulty": item.get("difficulty")}
        t_item = time.perf_counter()
        try:
            result = await run_agent_harness(
                objective=item["question"], tenant_id="eval", user_id="eval",
                user_context={"roles": ["researcher"]},
            )
            answer = result.final_answer or ""
            evidence = result.evidence or []
            passed, info = judge_answer(item, answer, evidence)
            row.update(passed=passed, status=result.status,
                       tool_calls=result.total_tool_calls,
                       loop_detected=any(a.get("loop_detected") for a in (result.audit_trace or [])),
                       hitl=result.human_review_required,
                       latency_ms=round((time.perf_counter() - t_item) * 1000, 1),
                       answer_preview=answer[:160], **info)
        except Exception as e:  # noqa: BLE001
            row.update(passed=None, error=f"{type(e).__name__}: {str(e)[:120]}")
        rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_fh.flush()
        all_rows.append(row)
        done = i + 1
        el = time.perf_counter() - t0
        eta = (el / done) * (len(items) - done)
        write_progress(phase="agent_eval", done=done, total=len(items),
                       percent=round(100 * done / len(items), 1),
                       elapsed_s=round(el, 1), eta_s=round(eta, 1), current_id=item["id"])
        print(f"[{done}/{len(items)}] {item['id']} ({row.get('difficulty')}) "
              f"pass={row.get('passed')} tool_calls={row.get('tool_calls')} "
              f"| {el:.0f}s eta {eta:.0f}s", flush=True)
    return all_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="L1 · Agent 端到端评测（跑 harness+LLM，判最终答案）")
    ap.add_argument("--categories", default="D,E,F", help="逗号分隔，默认 D,E,F（差异化能力所在）")
    ap.add_argument("--per-category", type=int, default=6, help="每类抽样题数（控规模，端到端慢）")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    log_fh = _attach_log()
    if not SET_PATH.is_file():
        print(f"❌ 评测集不存在：{SET_PATH}", file=sys.stderr)
        return 1
    allitems = [json.loads(line) for line in SET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    rng = random.Random(args.seed)
    items = []
    for c in cats:
        pool = [it for it in allitems if it["category"] == c]
        rng.shuffle(pool)
        items.extend(pool[: args.per_category])
    print(f"端到端评测：类={cats} 每类≤{args.per_category} → 共 {len(items)} 题（真跑 harness+LLM，慢）", flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    resume_rows = []
    if args.resume and ROWS_PATH.is_file():
        resume_rows = [json.loads(x) for x in ROWS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
        done_ids = {r["id"] for r in resume_rows}
        items = [it for it in items if it["id"] not in done_ids]
        print(f"--resume：已完成 {len(resume_rows)} 题，剩余 {len(items)}", flush=True)
    rows_fh = ROWS_PATH.open("a" if args.resume else "w", encoding="utf-8")

    all_rows = asyncio.run(_run_all(items, rows_fh, resume_rows))
    rows_fh.close()

    summary = _summarize(all_rows)
    print("\n=== Agent 端到端结果（按难度）===", flush=True)
    for k, v in sorted(summary["per_difficulty"].items()):
        print(f"  {k}: acc={v['accuracy']} ({v['pass']}/{v['judged']}, skip {v['skip']})", flush=True)
    print(f"  grounding 通过率: {summary['grounding_rate']} (n={summary['grounded_n']})", flush=True)

    out = REPORT_DIR / "agent_eval_report.json"
    out.write_text(json.dumps({"summary": summary, "rows": all_rows,
                               "config": {"categories": cats, "per_category": args.per_category,
                                          "currency": True, "query_rewrite": "off"}},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    write_progress(phase="done", done=len(all_rows), total=len(all_rows), percent=100,
                   report=str(out.relative_to(ROOT)))
    print(f"\n报告: {out.relative_to(ROOT)}｜逐题: {ROWS_PATH.relative_to(ROOT)}", flush=True)
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
