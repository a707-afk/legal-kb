# -*- coding: utf-8 -*-
"""BM25 单路探针：直接看 bm25_search 的原始 top-N（不经过融合/精排）。

用途：判断 BM25 候选是否被短 FAQ 文档（LawBench 基准题）刷屏。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QUERY_REWRITE_MODE", "off")

from src.config_LEGACY_REFERENCE import get_settings  # noqa: E402


def main() -> int:
    q = sys.argv[1] if len(sys.argv) > 1 else "《中华人民共和国农业法》现在生效了吗？什么时候施行？"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    s = get_settings()
    from src.bm25_store import _get_bm25, bm25_search, node_with_score_from_bm25
    path = getattr(s, "bm25_corpus_path_cn", None) or s.bm25_corpus_path
    if not Path(path).is_file():
        path = s.bm25_corpus_path
    print(f"bm25 corpus: {path}")
    hits = bm25_search(s, q, n, corpus_path=path)
    _, _, meta = _get_bm25(s, corpus_path=path)
    print(f"query: {q}\n")
    n_faq = 0
    for i, (nid, sc) in enumerate(hits):
        nws = node_with_score_from_bm25(nid, sc, meta)
        md = dict(getattr(getattr(nws, "node", None), "metadata", None) or {}) if nws else {}
        f = Path(str(md.get("file_name") or "")).name
        grp = str(md.get("doc_group") or md.get("source_type") or "")
        is_faq = "faq" in grp or "是什么" in f or "的内容" in f
        n_faq += 1 if is_faq else 0
        print(f"{i+1:>3} {sc:8.3f}  [{grp or '?'}] {f}")
    print(f"\nFAQ 污染: {n_faq}/{len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())