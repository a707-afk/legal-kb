# -*- coding: utf-8 -*-
"""把 OLE2 旧版 .doc 转成 .docx（一次性预处理）。

为什么需要这个脚本
------------------
`data/raw/npc/*/files/` 里有 63 个文件是**真正的 Word 97-2003 二进制格式**
（magic `D0 CF 11 E0`），`python-docx` **无法读取**。若不处理，这 63 篇法条会静默丢失
（司法解释一栏就少 58 篇）。

方案：用本机已安装的 Microsoft Word 做 COM 批量转换（`Visible=False`，无界面），
输出到 `data/raw/npc/_doc_converted/<category>/`。**原文件不改动。**

用法：
    python scripts/convert_doc_to_docx.py            # 转换全部缺失的
    python scripts/convert_doc_to_docx.py --limit 1  # 试跑一个
    python scripts/convert_doc_to_docx.py --dry-run

幂等：已存在同名 .docx 则跳过。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "npc"
OUT_ROOT = RAW / "_doc_converted"

WD_FORMAT_DOCX = 16  # wdFormatXMLDocument


def find_doc_files(categories: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for cat in categories:
        files_dir = RAW / cat / "files"
        if not files_dir.is_dir():
            continue
        for p in sorted(files_dir.glob("*.doc")):
            if p.suffix.lower() == ".doc":
                out.append((cat, p))
    return out


def convert_one(word, src: Path, dst: Path) -> None:
    """单个转换。失败抛异常，由调用方记录。"""
    doc = None
    try:
        doc = word.Documents.Open(str(src), ReadOnly=True, AddToRecentFiles=False)
        doc.SaveAs2(str(dst), FileFormat=WD_FORMAT_DOCX)
    finally:
        if doc is not None:
            doc.Close(SaveChanges=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="OLE2 .doc -> .docx 批量转换（Word COM）")
    ap.add_argument("--limit", type=int, default=0, help="最多转换几个（0=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只列出待转换文件")
    args = ap.parse_args()

    targets = find_doc_files(["法律", "行政法规", "司法解释"])
    todo: list[tuple[str, Path, Path]] = []
    skipped = 0
    for cat, src in targets:
        dst = OUT_ROOT / cat / (src.stem + ".docx")
        if dst.exists():
            skipped += 1
            continue
        todo.append((cat, src, dst))

    print(f"发现 .doc 文件 {len(targets)} 个；已转换 {skipped} 个；待转换 {len(todo)} 个")
    if args.dry_run:
        for cat, src, _ in todo[:20]:
            print(f"  [{cat}] {src.name}")
        if len(todo) > 20:
            print(f"  ... 另有 {len(todo) - 20} 个")
        return 0
    if not todo:
        print("无需转换。")
        return 0

    if args.limit:
        todo = todo[: args.limit]

    try:
        import win32com.client  # type: ignore
    except ImportError:
        print("缺少 pywin32（win32com）。请先 pip install pywin32", file=sys.stderr)
        return 2

    for cat, _, dst in todo:
        dst.parent.mkdir(parents=True, exist_ok=True)

    word = None
    ok = failed = 0
    failures: list[dict] = []
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        for i, (cat, src, dst) in enumerate(todo, 1):
            try:
                convert_one(word, src, dst)
                ok += 1
                print(f"  [{i}/{len(todo)}] OK   {src.name}")
            except Exception as exc:  # 单个失败不中断整体
                failed += 1
                failures.append({"category": cat, "file": src.name, "error": str(exc)})
                print(f"  [{i}/{len(todo)}] FAIL {src.name}: {exc}")
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass

    print(json.dumps(
        {"converted": ok, "failed": failed, "output_dir": str(OUT_ROOT)},
        ensure_ascii=False, indent=1,
    ))
    if failures:
        report = ROOT / "reports" / "doc_conversion_failures.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(failures, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"失败明细已写入: {report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
