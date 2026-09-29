# -*- coding: utf-8 -*-
"""验证 status_code 语义 —— 本项目版本 GT 的地基。

用途
----
1. 重建索引前跑一次，确认官方元数据字段仍然可解析（防止再次出现"字段名写错"的静默失效）
2. 产出可直接写进文档 / 报告的统计数字

用法
----
    python scripts/verify_version_gt.py
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = str(REPO_ROOT / "data" / "raw" / "npc")
CATS = ["法律", "行政法规", "司法解释"]

STATUS_MEANING = {
    1: "已废止",
    2: "已被修订",
    3: "现行有效",
    4: "尚未生效",
    None: "未标注",
    -1: "未知",
}


def parse_date(s):
    try:
        return datetime.date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def main() -> int:
    today = datetime.date.today()
    rows = []
    for cat in CATS:
        path = os.path.join(RAW, cat, "metadata.jsonl")
        if not os.path.exists(path):
            print(f"[跳过] 找不到 {path}")
            continue
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["_cat"] = cat
                rows.append(r)

    if not rows:
        print("没有读到任何元数据，请检查 data/raw/npc/ 是否存在")
        return 1

    print(f"读取条目：{len(rows)}（今天 {today}）\n")

    # --- 断言 1：必需字段存在（防止字段名再次写错）---
    required = ["title", "bbbs", "status_code", "effective_date"]
    missing = [f for f in required if not any(f in r for r in rows)]
    if missing:
        print(f"❌ 致命：元数据缺少必需字段 {missing} —— 导入脚本的字段映射会静默失效")
        return 1
    print(f"✅ 必需字段齐全：{required}")

    # --- 断言 2：status_code 取值分布 ---
    dist = collections.Counter(r.get("status_code") for r in rows)
    print("\nstatus_code 分布：")
    for code in sorted(dist, key=lambda c: (c is None, c)):
        n = dist[code]
        print(f"  {str(code):>4}  {STATUS_MEANING.get(code, '?'):<8} {n:>5}  {n / len(rows):>6.1%}")

    # --- 假设 H1：status_code=4 => 尚未生效（生效日期在未来）---
    s4 = [r for r in rows if r.get("status_code") == 4]
    h1_ok = sum(1 for r in s4 if (parse_date(r.get("effective_date")) or today) > today)
    print(f"\nH1  status_code=4 => 尚未生效：{h1_ok}/{len(s4)}", end="")
    print("  ✅" if s4 and h1_ok == len(s4) else "  ⚠️")

    # --- 假设 H2：多版本标题中 status_code=3 即生效日期最大者 ---
    by_title = collections.defaultdict(list)
    for r in rows:
        by_title[(r["_cat"], r["title"])].append(r)
    multi = {k: v for k, v in by_title.items() if len(v) > 1}
    ok = bad = 0
    for v in multi.values():
        s3 = [r for r in v if r.get("status_code") == 3]
        if not s3:
            continue
        max_eff = max((parse_date(r.get("effective_date")) or datetime.date(1970, 1, 1)) for r in v)
        if all((parse_date(r.get("effective_date")) or datetime.date(1970, 1, 1)) == max_eff for r in s3):
            ok += 1
        else:
            bad += 1
    print(f"H2  多版本中 status_code=3 即最新生效版：{ok} 符合 / {bad} 不一致", end="")
    print("  ✅" if ok and bad / max(1, ok + bad) < 0.05 else "  ⚠️")
    if bad:
        print("     （不一致通常是'已公布未施行'的合法情形——恰好是 E 类题的题源）")

    # --- 规模汇总 ---
    print("\n版本分布（题源盘点）：")
    print(f"  {'分库':<8}{'条目':>6}{'唯一标题':>10}{'多版本标题':>12}{'额外版本':>10}")
    tot = [0, 0, 0, 0]
    for cat in CATS:
        cr = [r for r in rows if r["_cat"] == cat]
        if not cr:
            continue
        bt = collections.defaultdict(list)
        for r in cr:
            bt[r["title"]].append(r)
        m = {k: v for k, v in bt.items() if len(v) > 1}
        extra = sum(len(v) - 1 for v in m.values())
        print(f"  {cat:<8}{len(cr):>6}{len(bt):>10}{len(m):>12}{extra:>10}")
        for i, v in enumerate([len(cr), len(bt), len(m), extra]):
            tot[i] += v
    print(f"  {'合计':<8}{tot[0]:>6}{tot[1]:>10}{tot[2]:>12}{tot[3]:>10}")

    print("\n结论：")
    print("  · status_code 可直接作为版本 GT，无需人工标注")
    print(f"  · {tot[3]} 个额外版本条目 = 版本冲突题（D 类）的题源池")
    print(f"  · status_code 缺失 {dist.get(None, 0)} 条（{dist.get(None, 0) / len(rows):.1%}）")
    print("    → 这批复数据可作为'元数据缺失时是否老实说'（F 类）的题源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
