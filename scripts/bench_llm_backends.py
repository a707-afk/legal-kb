# -*- coding: utf-8 -*-
"""生成层候选模型对照（延迟 + 引用精度）。

为什么需要这个脚本
------------------
`docs/06` 第四节的模型选型数字（glm-4.5-flash 33.8s vs glm-4-flash 1.0s）是
**2026-09-28 单次实测**，样本量为 1，且当时未纳入商汤后端。
D4/D5 的评测总时长直接由"单题生成延迟 × 题数 × 迭代次数"决定，必须用**可复跑的对照**定下来。

判据（本项目的核心指标是引用精度，见 D-07）
------------------------------------------
- **条号命中率**：答案里出现的 `第X条` 有多少比例能在给定片段中找到
- **项号精度**：答案是否精确到"第（二）项"（法条含多项时，只给条号等于没定位）
- 延迟 p50

用法
----
    export ZHIPU_API_KEY=...
    export SENSENOVA_API_KEY=...
    python scripts/bench_llm_backends.py                    # 默认 6 题 × 4 模型
    python scripts/bench_llm_backends.py --models glm-4.5-flash sensenova-6.8-flash-lite
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "llm_backend_bench.json"

SYS = ("你是一个法律合规分析器。仅依据提供的法条片段回答问题，并给出条号级引用。"
       "证据不足时明确说明，不得编造。")

# 每题：(问题, 片段列表)。片段里刻意包含"多项"的条文，用来区分"只给条号"与"给到项号"。
CASES: list[tuple[str, list[str]]] = [
    (
        "公司拖欠工资时，劳动者可以解除劳动合同吗？",
        [
            "《中华人民共和国劳动合同法》第三十八条：用人单位有下列情形之一的，劳动者可以解除劳动合同："
            "（一）未按照劳动合同约定提供劳动保护或者劳动条件的；（二）未及时足额支付劳动报酬的；"
            "（三）未依法为劳动者缴纳社会保险费的；（四）用人单位的规章制度违反法律、法规的规定，损害劳动者权益的；",
            "《中华人民共和国劳动合同法》第四十六条：有下列情形之一的，用人单位应当向劳动者支付经济补偿："
            "（一）劳动者依照本法第三十八条规定解除劳动合同的；（二）用人单位依照本法第三十六条规定向劳动者"
            "提出解除劳动合同并与劳动者协商一致解除劳动合同的；",
        ],
    ),
    (
        "买卖合同没有约定违约金，一方违约造成损失怎么赔？",
        [
            "《中华人民共和国民法典》第五百七十七条：当事人一方不履行合同义务或者履行合同义务不符合约定的，"
            "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
            "《中华人民共和国民法典》第五百八十四条：当事人一方不履行合同义务或者履行合同义务不符合约定，"
            "造成对方损失的，损失赔偿额应当相当于因违约所造成的损失，包括合同履行后可以获得的利益；"
            "但是，不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。",
        ],
    ),
    (
        "民事诉讼时效一般是多少年？",
        [
            "《中华人民共和国民法典》第一百八十八条：向人民法院请求保护民事权利的诉讼时效期间为三年。"
            "法律另有规定的，依照其规定。诉讼时效期间自权利人知道或者应当知道权利受到损害以及义务人之日起计算。",
        ],
    ),
    (
        "用人单位违法解除劳动合同要付多少赔偿金？",
        [
            "《中华人民共和国劳动合同法》第四十七条：经济补偿按劳动者在本单位工作的年限，每满一年支付一个月"
            "工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资"
            "的经济补偿。",
            "《中华人民共和国劳动合同法》第八十七条：用人单位违反本法规定解除或者终止劳动合同的，应当依照"
            "本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
        ],
    ),
    (
        "交通事故造成他人死亡的赔偿项目有哪些？",
        [
            "《中华人民共和国民法典》第一千一百七十九条：侵害他人造成人身损害的，应当赔偿医疗费、护理费、"
            "交通费、营养费、住院伙食补助费等为治疗和康复支出的合理费用，以及因误工减少的收入。造成残疾的，"
            "还应当赔偿辅助器具费和残疾赔偿金；造成死亡的，还应当赔偿丧葬费和死亡赔偿金。",
        ],
    ),
    (
        "承租人擅自转租，出租人可以解除合同吗？",
        [
            "《中华人民共和国民法典》第七百一十六条：承租人经出租人同意，可以将租赁物转租给第三人。"
            "承租人转租的，承租人与出租人之间的租赁合同继续有效；第三人造成租赁物损失的，承租人应当赔偿损失。"
            "承租人未经出租人同意转租的，出租人可以解除合同。",
        ],
    ),
]

from src.legal_refs import article_hit, extract_citations, normalize_article_text  # noqa: E402


def _norm_cn(s: str) -> str:
    return re.sub(r"[\s　]+", "", s)


def score(answer: str, snippets: list[str]) -> dict:
    """引用精度判分（确定性规则，不用 LLM 打分）。

    ⚠️ 必须走 `src.legal_refs` 的**中文数字归一化**：模型常写 `第577条`，
    而语料原文是 `第五百七十七条`。不做归一化会把正确引用判成幻觉
    （实测把 `sensenova-6.8-flash-lite` 的条号精度从 1.0 误判为 0.781）。
    """
    ctx = "".join(snippets)
    cits = extract_citations(answer)
    hit, total = article_hit(cits, ctx)

    ctx_norm = normalize_article_text(_norm_cn(ctx))
    items = [c for c in cits if c.item is not None]
    item_hit = sum(
        1 for c in items
        if f"（{c.item}）" in ctx_norm or f"第{c.item}{c.item_kind or '项'}" in ctx_norm
    )
    return {
        "articles_cited": total,
        "articles_hit": hit,
        "article_precision": round(hit / total, 3) if total else None,
        "items_cited": len(items),
        "items_hit": item_hit,
        "item_precision": round(item_hit / len(items), 3) if items else None,
        "answer_chars": len(answer),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="生成层候选模型对照")
    ap.add_argument("--models", nargs="*", default=[
        "sensenova-6.8-flash-lite", "deepseek-v4-flash",
    ])
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 题（0=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只列将调用的模型与题数，不真调")
    args = ap.parse_args()

    cases = CASES[: args.limit] if args.limit else CASES
    n_calls = len(args.models) * len(cases)
    print(f"将调用 {len(args.models)} 个模型 × {len(cases)} 题 = {n_calls} 次")
    if args.dry_run:
        for m in args.models:
            print(f"  - {m}")
        return 0

    from src.llm import chat_completion_full, get_key_pool_size
    import src.llm as llm_mod

    print(f"凭据池大小: {get_key_pool_size()} 个 key（逗号分隔自动轮转）")

    results: dict[str, dict] = {}
    for model in args.models:
        # 单后端（SenseNova）：只覆盖模型名做对照，不再有后端分派
        llm_mod.get_model_name = lambda m=model: m
        latencies: list[float] = []
        rows: list[dict] = []
        errors = 0
        for i, (q, snips) in enumerate(cases, 1):
            user = f"法律问题：{q}\n\n证据片段：\n" + "\n".join(
                f"[{j}] {s}" for j, s in enumerate(snips, 1)
            ) + "\n\n请生成带条号引用的回答："
            try:
                out = chat_completion_full(SYS, user, temperature=0.1)
            except Exception as exc:
                errors += 1
                print(f"  [{model}] {i}/{len(cases)} ❌ {type(exc).__name__}: {str(exc)[:120]}")
                continue
            sc = score(out["content"], snips)
            latencies.append(out["latency_ms"])
            rows.append({"q": q[:30], **sc, "latency_ms": out["latency_ms"],
                         "attempts": out["attempts"]})
            print(f"  [{model}] {i}/{len(cases)} {out['latency_ms']/1000:6.2f}s "
                  f"条号 {sc['articles_hit']}/{sc['articles_cited']} "
                  f"项号 {sc['items_hit']}/{sc['items_cited']} {sc['answer_chars']}字")

        art_ok = [r["articles_hit"] / r["articles_cited"] for r in rows if r["articles_cited"]]
        item_ok = [r["items_hit"] / r["items_cited"] for r in rows if r["items_cited"]]
        results[model] = {
            "backend": llm_mod.BACKEND_NAME,
            "n_ok": len(rows),
            "n_errors": errors,
            "latency_p50_s": round(statistics.median(latencies) / 1000, 2) if latencies else None,
            "latency_mean_s": round(statistics.mean(latencies) / 1000, 2) if latencies else None,
            "article_precision_mean": round(statistics.mean(art_ok), 3) if art_ok else None,
            "item_precision_mean": round(statistics.mean(item_ok), 3) if item_ok else None,
            "cites_item_at_all": sum(1 for r in rows if r["items_cited"]),
            "rows": rows,
        }
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_cases": len(cases), "results": results},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 78)
    print(f"{'模型':28s} {'延迟p50':>8s} {'条号精度':>9s} {'项号精度':>9s} {'给出项号题数':>12s}")
    print("-" * 78)
    for m, r in results.items():
        print(f"{m:28s} {str(r['latency_p50_s'])+'s':>8s} "
              f"{str(r['article_precision_mean']):>9s} {str(r['item_precision_mean']):>9s} "
              f"{r['cites_item_at_all']:>7d}/{r['n_ok']}")
    print(f"\n报告: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
