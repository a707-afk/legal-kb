# -*- coding: utf-8 -*-
"""构建 Qdrant 向量索引 + BM25 语料（D2）。

流程
----
1. 加载 `data/docs/**.md` → 切块（`chunk_strategy=hierarchical_recursive`）
2. **重复节点自检**（全文 MD5）——目标 0 个字节级重复
   （旧项目"切块函数零前进循环"曾让 495 篇判决书各产生约 1,500 个重复节点 → MemoryError）
3. **维度断言**：模型维度 == `EMBEDDING_DIMENSION` == 集合维度，不一致直接抛错
4. 重建集合（先 delete 再建，1024 维 cosine）+ 写入 BM25 语料
5. 核对集合点数

用法
----
    python scripts/build_index.py --dry-run        # 只切块 + 自检，不建库
    python scripts/build_index.py                  # 全量重建（向量落盘，可断点续跑）
    python scripts/build_index.py --keep-existing  # 不删集合（增量）
    python scripts/build_index.py --fresh-cache    # 作废旧向量缓存，全部重编
    python scripts/build_index.py --batch-size 128 # 覆盖 EMBEDDING_BATCH_SIZE

可观测性（两个出口，不用盯终端）
------------------------------
- 日志：`logs/build_index.log`（逐行 tee，UTF-8）
- 进度：`data/.embed_cache/progress.json`（done / total / 速率 / ETA / 阶段）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports" / "build_index_report.json"
LOG_PATH = ROOT / "logs" / "build_index.log"


class _Tee:
    """同时写终端和日志文件。

    分两条路各自编码：终端沿用原生编码（并把不可编码字符换成 ?，避开 GBK
    碰上 emoji 直接 UnicodeEncodeError 崩掉整个建库进程——已在 verify_ingest.py
    上踩过），日志文件固定 UTF-8 保留完整文本。
    """

    def __init__(self, stream, fh) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:  # noqa: BLE001  终端炸不能影响主流程
            pass
        self._fh.write(data)
        self._fh.flush()
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


def _attach_log() -> object:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = LOG_PATH.open("a", encoding="utf-8", errors="replace")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")  # 保编码、不崩
        except Exception:  # noqa: BLE001
            pass
        setattr(sys, name, _Tee(stream, fh))
    return fh



def duplicate_check(nodes) -> dict:
    """按**全文 MD5** 找重复节点，并区分两类性质完全不同的重复。

    不要用"前 100 字前缀"当判据：旧项目审查时用前缀判重复得 3,794，
    改用全文 MD5 后是 226，**差 16 倍**（HANDOFF §9.4：截断输出不能当证据）。

    两类重复（**判据是三元组：文件 + heading_path + 文本**）：
    - **同位置重复（`intra_doc_extra`）**：同一文件、同一 heading_path 下出现内容完全相同的节点
      → 这才是旧项目"切块零前进循环"那类 **bug** 的指标，验收要求 = 0
    - **跨文档重复（`cross_doc_extra`）**：不同文件里有相同文本
      → **预期现象**：本语料保留 377 个多版本标题（如《人民检察院刑事诉讼规则》
        与《…（试行）》正文高度相同），加上"本法自公布之日起施行"这类通用条款。
        要求为 0 等于要求删掉多版本语料，与 D-01 直接冲突

    为什么不按"纯文本"判重：一份文件含多个附件时，各附件**编号重新从第一条开始**，
    文字可能完全相同（实测 `重新组建仲裁机构方案` 的附件一/附件二第二条）。
    那是源文档的真实结构，不是重复——用 heading_path 区分后 `intra_doc_extra` 为 0。
    """
    triples: Counter = Counter()
    texts: dict[str, list] = {}
    for n in nodes:
        meta = n.metadata or {}
        text = (n.get_content() or "").strip()
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        triples[(str(meta.get("file_path") or ""), str(meta.get("heading_path") or ""), digest)] += 1
        texts.setdefault(digest, []).append(n)

    intra_extra = 0
    intra_examples: list[str] = []
    for (fp, hp, _digest), cnt in triples.items():
        if cnt > 1:
            intra_extra += cnt - 1
            if len(intra_examples) < 5:
                intra_examples.append(f"{Path(fp).name} | {hp[:40]} | x{cnt}")

    cross_extra = 0
    dup_groups = 0
    worst_group = 1
    for digest, members in texts.items():
        if len(members) < 2:
            continue
        dup_groups += 1
        worst_group = max(worst_group, len(members))
        by_file = {str((m.metadata or {}).get("file_path") or "") for m in members}
        cross_extra += len(by_file) - 1

    return {
        "unique_digests": len(texts),
        "duplicate_groups": dup_groups,
        "intra_doc_extra": intra_extra,
        "cross_doc_extra": cross_extra,
        "worst_group_size": worst_group,
        "intra_doc_examples": intra_examples,
        "key": "file_path + heading_path + text_md5",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 Qdrant 索引 + BM25 语料")
    ap.add_argument("--dry-run", action="store_true", help="只切块与自检，不建库")
    ap.add_argument("--keep-existing", action="store_true", help="不删已有集合")
    ap.add_argument("--no-cache", action="store_true", help="不用向量缓存（全部重算且不落盘）")
    ap.add_argument("--fresh-cache", action="store_true",
                    help="作废旧向量缓存（移至 .stale-<ts>，不静默删）")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="覆盖 EMBEDDING_BATCH_SIZE（8GB 显存上建议 64–128）")
    args = ap.parse_args()

    log_fh = _attach_log()
    print(f"\n===== build_index 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
    print(f"日志: {LOG_PATH.relative_to(ROOT)}", flush=True)

    from src.chunking import build_nodes, load_documents
    from src.config_LEGACY_REFERENCE import get_settings

    settings = get_settings()
    if args.batch_size:
        settings.embedding_batch_size = int(args.batch_size)
    docs_dir = Path(settings.docs_dir).resolve()
    print(f"语料目录: {docs_dir}")
    print(f"集合: {settings.qdrant_collection_name} ｜ "
          f"qdrant={'path:' + str(settings.qdrant_path) if settings.qdrant_path and not settings.qdrant_url else 'url:' + str(settings.qdrant_url)}")
    print(f"切块策略: {settings.chunk_strategy} (chunk={settings.chunk_size_tokens}/overlap={settings.chunk_overlap_tokens})")
    print(f"编码配置: batch={settings.embedding_batch_size} "
          f"fp16={settings.embedding_fp16} 期望维度={settings.embedding_dimension}")
    print(f"向量缓存: {settings.embed_cache_dir or '(关闭)'} ｜ 进度文件: "
          f"{settings.build_progress_path or '(关闭)'}")
    print()

    t0 = time.perf_counter()
    documents = load_documents(docs_dir)
    print(f"[1/5] 加载文档: {len(documents)} 篇  ({time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    nodes = build_nodes(documents, settings)
    print(f"[2/5] 切块: {len(nodes)} 个节点  ({time.perf_counter()-t1:.1f}s)")

    t2 = time.perf_counter()
    dup = duplicate_check(nodes)
    print(f"[3/5] 重复自检（全文 MD5）: 唯一 {dup['unique_digests']} ｜ 重复组 {dup['duplicate_groups']}")
    print(f"      同文档内重复（验收=0）: {dup['intra_doc_extra']} ｜ "
          f"跨文档相同文本（预期，多版本语料）: {dup['cross_doc_extra']} ｜ "
          f"最大组 {dup['worst_group_size']}  ({time.perf_counter()-t2:.1f}s)")
    if dup["intra_doc_examples"]:
        for ex in dup["intra_doc_examples"][:3]:
            print(f"      ⚠️ {ex[:100]}")

    # 时效元数据是否随节点带下来（D-02/D-06 的前提）
    with_sc = sum(1 for n in nodes if "status_code" in (n.metadata or {}))
    with_eff = sum(1 for n in nodes if (n.metadata or {}).get("effective_date"))
    print(f"      时效元数据覆盖: status_code {with_sc}/{len(nodes)} ｜ effective_date {with_eff}/{len(nodes)}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "docs_dir": str(docs_dir),
        "collection": settings.qdrant_collection_name,
        "qdrant_mode": "path" if (settings.qdrant_path and not settings.qdrant_url) else "url",
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "documents": len(documents),
        "nodes": len(nodes),
        "duplicates": dup,
        "nodes_with_status_code": with_sc,
        "nodes_with_effective_date": with_eff,
        "chunk_seconds": round(time.perf_counter() - t1, 1),
    }

    if args.dry_run:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n(dry-run) 报告: {REPORT.relative_to(ROOT)}")
        return 0

    # ── 建库 ──
    from src.embeddings import get_embedding_model
    from src.vector_store import get_qdrant_client, rebuild_index

    t3 = time.perf_counter()
    # 维度断言：模型维度 == EMBEDDING_DIMENSION，不一致直接抛错（不静默回退）
    dim = get_embedding_model().assert_dimension(settings.embedding_dimension)
    print(f"[4/5] embedding 模型就绪: 维度 {dim}（已断言）  ({time.perf_counter()-t3:.1f}s)")

    t4 = time.perf_counter()
    n_indexed = rebuild_index(
        use_cache=False if args.no_cache else None,
        fresh_cache=args.fresh_cache,
    )
    print(f"[5/5] 索引完成: {n_indexed} 个节点  ({time.perf_counter()-t4:.1f}s)")

    client = get_qdrant_client(settings)
    info = client.get_collection(settings.qdrant_collection_name)
    points = info.points_count
    vectors = info.config.params.vectors
    coll_dim = getattr(vectors, "size", None) or (
        next(iter(v.size for v in vectors.values())) if isinstance(vectors, dict) else None
    )
    print(f"      集合点数: {points} ｜ 维度: {coll_dim}")

    report.update({
        "embedding_dimension": dim,
        "collection_points": points,
        "collection_dimension": coll_dim,
        "index_seconds": round(time.perf_counter() - t4, 1),
        "total_seconds": round(time.perf_counter() - t0, 1),
    })
    # 分段计时与缓存命中（来自 src/vector_store.last_rebuild_stats）
    try:
        from src.vector_store import last_rebuild_stats

        report["embed_stats"] = dict(last_rebuild_stats)
    except Exception:  # noqa: BLE001
        report["embed_stats"] = {}
    report["acceptance"] = {
        "points_ge_30000": bool(points and points >= 30000),
        "no_intra_doc_duplicates": dup["intra_doc_extra"] == 0,
        "dimension_match": dim == coll_dim,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(json.dumps(report["acceptance"], ensure_ascii=False, indent=1))
    print(f"报告: {REPORT.relative_to(ROOT)}")
    print(f"日志: {LOG_PATH.relative_to(ROOT)}")
    if report["embed_stats"].get("cache_hits"):
        print(
            f"本次缓存命中 {report['embed_stats']['cache_hits']} 个节点"
            f"（新增编码 {report['embed_stats'].get('nodes_to_embed', 0) - report['embed_stats']['cache_hits']}）"
        )

    # 显式关闭 Qdrant 客户端（本地 path 模式靠 close 刷盘）。
    # 不关则依赖解释器退出时的 __del__，而那时 sys.meta_path 已为 None →
    # 抛 "ImportError: sys.meta_path is None"，刷盘不可靠且进程迟迟不退出
    # （实测本次建库就是打完全部日志后卡着不结束，很容易被误判成还在跑）。
    try:
        from src.vector_store import clear_index_memory_cache

        clear_index_memory_cache()
        print("Qdrant 客户端已显式关闭（本地模式刷盘完成）")
    except Exception as e:  # noqa: BLE001
        print(f"关闭 Qdrant 客户端异常（不影响已写入数据）: {type(e).__name__}: {e}")

    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
