# -*- coding: utf-8 -*-
"""D1 验收校验：`data/docs/` 与 `data/raw/npc/` 元数据逐条对齐。

验收口径（PLAN.md D1）
--------------------
1. `data/docs/` 文档数 ≥ 2,400
2. 随机抽 5 篇，`status_code` 与官网元数据一致
3. 多版本标题数 = 377（±5）
4. 剔除的条目确实是"无正文占位条目"（正文 < 100 字）

另外校验（防止旧项目的两类事故重演）
----------------------------------
- **字段名事故**：front-matter 里的 `status` 必须是映射结果，不能是硬编码 `current`
- **静默覆盖事故**：文件名全局唯一
- **目录污染**：正文里的"目录"块应已被剥掉

用法：`python scripts/verify_ingest.py [--sample 5] [--seed 42]`
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "npc"
DOCS = ROOT / "data" / "docs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CATEGORIES = ("法律", "行政法规", "司法解释")
DOMAIN_OF = {"法律": "statute", "行政法规": "statute", "司法解释": "interpretation"}
EXPECTED_MULTI_VERSION = 377
TOLERANCE = 5
MIN_DOCS = 2400
MIN_BODY_CHARS = 100

STATUS_LABEL = {
    3: "现行有效", 2: "已被修订", 1: "已废止", 4: "尚未生效",
    -1: "未知", None: "未标注",
}
RE_TIAO = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+条")
RE_TOC = re.compile(r"^目\s*录$")


def parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("'\"")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="D1 语料导入验收校验")
    ap.add_argument("--sample", type=int, default=5, help="抽样比对条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    failures: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        if not ok:
            failures.append(name)

    # ── 载入元数据，建 bbbs -> record 索引 ──
    meta_by_bbbs: dict[str, dict] = {}
    title_versions: Counter[str] = Counter()
    for cat in CATEGORIES:
        for line in (RAW / cat / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["_category"] = cat
            meta_by_bbbs[rec["bbbs"]] = rec
            title_versions[rec["title"]] += 1

    # ── 扫描产出 ──
    md_files = sorted(DOCS.rglob("*.md"))
    n_docs = len(md_files)

    # 验收 1：文档数
    check(f"1. 文档数 ≥ {MIN_DOCS}", n_docs >= MIN_DOCS, f"实际 {n_docs}")

    # 全局唯一性
    names = [p.name for p in md_files]
    dup_names = {k: v for k, v in Counter(names).items() if v > 1}
    check("2. 文件名全局唯一（无静默覆盖）", not dup_names, f"冲突 {len(dup_names)} 组")

    # 逐篇解析 front-matter 并与元数据比对
    matched = 0
    mismatch: list[str] = []
    status_hardcoded: list[str] = []
    empty_body: list[str] = []
    domain_counter: Counter[str] = Counter()
    sc_counter: Counter[int | None] = Counter()
    parsed: list[tuple[Path, dict, str]] = []
    by_version: dict[str, tuple[Path, dict, str]] = {}

    for p in md_files:
        text = p.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        body = text.split("---", 2)[-1] if text.startswith("---") else text
        domain_counter[fm.get("domain", "?")] += 1
        parsed.append((p, fm, body))
        if fm.get("version"):
            by_version[fm["version"]] = (p, fm, body)

        if len(re.sub(r"\s+", "", body)) < 10:
            empty_body.append(p.name)

        bbbs = fm.get("version", "")
        rec = meta_by_bbbs.get(bbbs)
        if rec is None:
            continue  # FAQ 等非 npc 来源
        matched += 1

        # 验收 2：status_code / effective_date / publish_date 与官网一致
        sc_fm = fm.get("status_code", "")
        sc_fm_val = None if sc_fm in ("", "null") else int(sc_fm)
        if sc_fm_val != rec.get("status_code"):
            mismatch.append(f"{p.name}: status_code fm={sc_fm!r} meta={rec.get('status_code')!r}")
        for field in ("effective_date", "publish_date"):
            if fm.get(field, "") != (rec.get(field) or ""):
                mismatch.append(f"{p.name}: {field} fm={fm.get(field)!r} meta={rec.get(field)!r}")
        # status 标签不能硬编码
        expect_label = STATUS_LABEL.get(rec.get("status_code"))
        if fm.get("status_label", "") != expect_label:
            status_hardcoded.append(f"{p.name}: label={fm.get('status_label')!r} 应为 {expect_label!r}")
        sc_counter[rec.get("status_code")] += 1

    check("3. front-matter 与官网元数据逐条一致", not mismatch,
          f"比对 {matched} 篇，不一致 {len(mismatch)}" + (f"；例: {mismatch[:3]}" if mismatch else ""))
    check("4. `status` 非硬编码（映射自 status_code）", not status_hardcoded,
          f"异常 {len(status_hardcoded)}" + (f"；例: {status_hardcoded[:3]}" if status_hardcoded else ""))
    check("5. 正文非空", not empty_body, f"空正文 {len(empty_body)}")

    # ── 验收 6（精确口径）：凡源文档含「目录」的，剥离后正文起始处必须是实质内容 ──
    # 只对 ingest 报告里记录了"剥离过目录"的文档做检查——这是真正能判定
    # "目录块有没有泄漏进正文"的判据。不用"重复短行"这类泛化启发式：
    # 官方司法解释天然含"公告抬头 + 正文落款"（两处都出现机关名），
    # 以及"罪名清单"这类本身就是连续短行的真实内容，泛化启发式会误报。
    report_path = ROOT / "reports" / "ingest_npc_report.json"
    toc_stripped_bbbs: list[str] = []
    if report_path.is_file():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        for v in rep.get("npc", {}).values():
            toc_stripped_bbbs.extend(v.get("toc_stripped_bbbs", []))
    toc_bad: list[str] = []
    toc_checked = 0
    for bbbs in toc_stripped_bbbs:
        entry = by_version.get(bbbs)
        if entry is None:
            toc_bad.append(f"{bbbs}: 报告标记剥离过目录，但产出缺失")
            continue
        p, _, body = entry
        toc_checked += 1
        content = [
            re.sub(r"[　\s*]+", "", ln.lstrip("#").strip())
            for ln in body.splitlines()
            if ln.strip() and not ln.startswith("#")
        ][:3]
        # 正文起始 3 行内必须出现条文或长段落
        if not any(RE_TIAO.match(c) or len(c) > 30 for c in content):
            toc_bad.append(f"{p.name}: 正文起始 {content[:2]}")
    check("6. 含「目录」的文档已剥离目录块（正文起始即实质内容）", not toc_bad,
          f"检查 {toc_checked} 篇，异常 {len(toc_bad)}"
          + (f"；例: {toc_bad[:3]}" if toc_bad else ""))

    # 验收 3：多版本标题数
    multi = sum(1 for _, v in title_versions.items() if v > 1)
    check(f"7. 多版本标题 = {EXPECTED_MULTI_VERSION}（±{TOLERANCE}）",
          abs(multi - EXPECTED_MULTI_VERSION) <= TOLERANCE, f"实际 {multi}")

    # 验收 4：被剔除的条目确实是占位条目
    kept_bbbs = {fm.get("version") for _, fm, _ in parsed}
    dropped = [
        (b, r) for b, r in meta_by_bbbs.items()
        if b not in kept_bbbs and r["_category"] in CATEGORIES
    ]
    # 逐一确认：这些条目的源文件正文确实 < 100 字
    from ingest_npc import _norm, docx_paragraphs, render_statute, resolve_source_file
    really_short = 0
    wrongly_dropped: list[str] = []
    for bbbs, rec in dropped:
        src = resolve_source_file(rec, rec["_category"])
        if src is None:
            wrongly_dropped.append(f"{bbbs} 文件缺失")
            continue
        try:
            body, _ = render_statute(docx_paragraphs(src), rec["title"])
        except Exception as exc:
            wrongly_dropped.append(f"{bbbs} 读取失败 {exc}")
            continue
        if len(_norm(body)) < MIN_BODY_CHARS:
            really_short += 1
        else:
            wrongly_dropped.append(f"{bbbs} 正文 {len(_norm(body))} 字 ≥ {MIN_BODY_CHARS} 却未入库")
    check("8. 剔除条目确为无正文占位（正文 < 100 字）",
          not wrongly_dropped,
          f"剔除 {len(dropped)} 篇，其中确认过短 {really_short}"
          + (f"；误剔 {len(wrongly_dropped)}: {wrongly_dropped[:3]}" if wrongly_dropped else ""))

    # ── 输出 ──
    print("=" * 66)
    print("D1 验收校验")
    print("=" * 66)
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"   — {detail}" if detail else ""))

    print()
    print(f"产出目录: {DOCS.relative_to(ROOT)}")
    print(f"  按 domain: {dict(domain_counter)}")
    print(f"  入库条目按 status_code: "
          f"{ {k: v for k, v in sorted(sc_counter.items(), key=lambda x: (x[0] is None, x[0]))} }")
    print(f"  与元数据成功对齐: {matched} 篇")

    # 验收 2 的抽样展示
    npc_parsed = [(p, fm) for p, fm, _ in parsed if fm.get("version") in meta_by_bbbs]
    rng = random.Random(args.seed)
    sample = rng.sample(npc_parsed, min(args.sample, len(npc_parsed)))
    print()
    print(f"抽样 {len(sample)} 篇（seed={args.seed}）status_code 比对：")
    for p, fm in sample:
        rec = meta_by_bbbs[fm["version"]]
        ok = (None if fm["status_code"] in ("", "null") else int(fm["status_code"])) == rec.get("status_code")
        print(f"  {'✅' if ok else '❌'} {p.name[:56]}")
        print(f"       官网 status_code={rec.get('status_code')!r} ({STATUS_LABEL.get(rec.get('status_code'))})"
              f" | effective={rec.get('effective_date') or '-'}"
              f" | front-matter={fm.get('status_code')} / {fm.get('status_label')}")

    print()
    if failures:
        print(f"❌ 未通过 {len(failures)}/{len(checks)} 项: {failures}", file=sys.stderr)
        return 1
    print(f"✅ 全部 {len(checks)} 项验收通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
