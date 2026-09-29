# -*- coding: utf-8 -*-
"""从语料构建 BM25 用的**法律术语自定义词典**（jieba 用户词典格式）。

为什么需要
----------
`src/bm25_store._tokenize_zh` 用 jieba 默认词典切词，法律术语会被切碎：
`诉讼时效` → `诉讼` / `时效`；`劳动合同法` → `劳动` / `合同` / `法`。
BM25 是**精确词项匹配**，切碎后既伤召回也伤精度（B 类口语化题尤其明显）。

术语来源（全部可程序抽取，不手工编）
--------------------------------
1. **法规标题**（`data/docs/**` 的 front-matter `title`）—— 法名是最高频的检索锚点
2. **章节标题**（正文里的 `编/章/节` 行）—— 如 `劳动合同的解除和终止`
3. **罪名**（司法解释里的罪名清单，形如 `1．背叛国家罪（第102条）`）
4. **法名派生**（去掉 `中华人民共和国` 前缀 + 补 `简称`）

输出
----
    data/dict/legal_terms.txt      jieba 用户词典（`词 词频 [词性]`）
    reports/legal_dict_report.json 统计

用法
----
    python scripts/build_legal_dict.py --dry-run
    python scripts/build_legal_dict.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOCS = ROOT / "data" / "docs"
OUT_DICT = ROOT / "data" / "dict" / "legal_terms.txt"
REPORT = ROOT / "reports" / "legal_dict_report.json"

RE_TITLE = re.compile(r'^title:\s*"?(.*?)"?\s*$')
RE_DOMAIN = re.compile(r"^domain:\s*\"?(.*?)\"?\s*$")
RE_HEADING = re.compile(r"^#{2,4}\s+(.+?)\s*$")
# 罪名：形如 `1．背叛国家罪（第102条）` 或 `背叛国家罪`
RE_CRIME_LINE = re.compile(r"^[\d一二三四五六七八九十]+[．.、]\s*(.{2,20}?罪)\s*[（(]")
RE_CN = "一二三四五六七八九十百千零〇"

# 只从这些 domain 取"标题"作术语。FAQ 的 title 是 LawBench 的**问句**
# （如 `社会法红十字会法第二十一条的内容是什么`），不是法律术语，必须排除。
TITLE_DOMAINS = {"statute", "interpretation"}

# 法名里必须剔除的噪音（不是术语）
STOPWORDS = {
    "中华人民共和国", "全国人民代表大会", "全国人民代表大会常务委员会",
    "最高人民法院", "最高人民检察院", "国务院", "人民法院", "人民检察院",
    "决定", "规定", "办法", "通知", "公告", "批复", "解释", "意见", "规则",
    "若干问题", "若干规定", "有关问题", "试行", "修正案",
}


def collect_terms(docs_dir: Path) -> tuple[Counter, dict]:
    """扫描语料，收集术语候选（带出现次数作为词频）。"""
    terms: Counter = Counter()
    stats = {"files": 0, "law_titles": 0, "headings": 0, "crimes": 0}

    for p in sorted(docs_dir.rglob("*.md")):
        stats["files"] += 1
        text = p.read_text(encoding="utf-8")
        title = ""
        domain = ""
        for line in text.splitlines()[:30]:
            if not title:
                m = RE_TITLE.match(line)
                if m:
                    title = m.group(1).strip()
            if not domain:
                md = RE_DOMAIN.match(line)
                if md:
                    domain = md.group(1).strip()
            if title and domain:
                break
        if title and domain in TITLE_DOMAINS:
            stats["law_titles"] += 1
            terms[title] += 50  # 法名权重高：它是最强的检索锚点
            # 去掉"中华人民共和国"前缀的简称
            short = title.replace("中华人民共和国", "").strip()
            if len(short) >= 4:
                terms[short] += 30

        body = text.split("---", 2)[-1] if text.startswith("---") else text
        for line in body.splitlines():
            mh = RE_HEADING.match(line)
            if mh:
                h = re.sub(r"[　\s]+", "", mh.group(1))
                h = re.sub(rf"^第[{RE_CN}0-9]+[编章节]\s*", "", h).strip()
                if 3 <= len(h) <= 30:
                    terms[h] += 5
                    stats["headings"] += 1
            mc = RE_CRIME_LINE.match(line.strip())
            if mc:
                terms[mc.group(1)] += 20
                stats["crimes"] += 1

    return terms, stats


def clean_terms(terms: Counter) -> Counter:
    """过滤噪音：停用词、过短/过长、含标点、纯数字、问句残留。"""
    out: Counter = Counter()
    for term, freq in terms.items():
        t = term.strip()
        if not t or t in STOPWORDS:
            continue
        if len(t) < 2 or len(t) > 40:
            continue
        if re.search(r"[，。；：、（）()《》〈〉\[\]【】\d]", t):
            continue
        if re.fullmatch(rf"[第{RE_CN}0-9编章节条款项]+", t):
            continue
        # 问句残留（FAQ 的 title 是问句，即使按 domain 过滤了也再兜一层）
        if re.search(r"(的内容是什么|是什么|有哪些|怎么办|如何|吗[？?]?$)", t):
            continue
        out[t] += freq
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="构建法律术语自定义词典（jieba 用户词典）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--min-freq", type=int, default=1, help="最低词频")
    ap.add_argument("--top", type=int, default=0, help="只保留词频最高的 N 个（0=全部）")
    args = ap.parse_args()

    if not DOCS.is_dir():
        print(f"语料目录不存在: {DOCS}", file=sys.stderr)
        return 2

    raw, stats = collect_terms(DOCS)
    terms = clean_terms(raw)
    terms = Counter({t: f for t, f in terms.items() if f >= args.min_freq})
    if args.top:
        terms = Counter(dict(terms.most_common(args.top)))

    report = {
        **stats,
        "terms_raw": len(raw),
        "terms_kept": len(terms),
        "top20": terms.most_common(20),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))

    if args.dry_run:
        return 0

    OUT_DICT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{t} {f}" for t, f in sorted(terms.items(), key=lambda x: (-x[1], x[0]))]
    OUT_DICT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n词典: {OUT_DICT.relative_to(ROOT)}  ({len(lines)} 条)")
    print(f"报告: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
