"""BM25 词法检索：与向量候选合并；语料在 reindex 时落盘。支持中英文双 BM25 语料。

性能：语料分词 + BM25Okapi 构建是分钟级一次性开销。索引按
（语料路径 + mtime + size）为键 pickle 持久化到 ``<语料目录>/.bm25_cache/``，
二次加载走反序列化（秒级），语料变更后自动失效重建。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import threading
import time
from pathlib import Path

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.schema import NodeWithScore
from rank_bm25 import BM25Okapi

from src.config_LEGACY_REFERENCE import Settings

logger = logging.getLogger(__name__)

_BM25_CACHE_VERSION = 1

_bm25: BM25Okapi | None = None
_bm25_ids: list[str] | None = None
_bm25_meta: dict[str, dict] | None = None

# 中文 BM25 独立缓存
_bm25_cn: BM25Okapi | None = None
_bm25_ids_cn: list[str] | None = None
_bm25_meta_cn: dict[str, dict] | None = None

# Thread safety for cache loading
_bm25_lock = threading.Lock()


def clear_bm25_memory_cache() -> None:
    global _bm25, _bm25_ids, _bm25_meta
    global _bm25_cn, _bm25_ids_cn, _bm25_meta_cn
    _bm25 = None
    _bm25_ids = None
    _bm25_meta = None
    _bm25_cn = None
    _bm25_ids_cn = None
    _bm25_meta_cn = None


def persist_bm25_corpus(nodes: list[BaseNode], settings: Settings, *, corpus_path: str | None = None) -> Path:
    """将节点写入 JSONL，供 BM25 加载。"""
    path = Path(corpus_path or settings.bm25_corpus_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n in nodes:
        nid = n.node_id
        text = n.get_content() or ""
        meta = dict(n.metadata or {})
        lines.append(
            json.dumps(
                {"id": nid, "text": text, "metadata": meta},
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("BM25 语料已写入 %s (%s 条)", path, len(lines))
    clear_bm25_memory_cache()
    return path


_dict_loaded: str | None = None


def _ensure_legal_dict() -> bool:
    """加载法律术语自定义词典（jieba 用户词典），只加载一次。

    为什么需要：jieba 默认词典会把法律术语切碎——`诉讼时效` → `诉讼`/`时效`，
    `劳动合同法` → `劳动`/`合同`/`法`。BM25 是精确词项匹配，切碎后既伤召回也伤精度。
    词典由 `scripts/build_legal_dict.py` 从语料构建（法名 + 章节标题 + 罪名）。

    返回是否加载成功；失败只记 warning 不抛（退回默认词典仍可用，但会伤效果）。
    """
    global _dict_loaded
    try:
        from src.config_LEGACY_REFERENCE import get_settings

        path = str(getattr(get_settings(), "bm25_dict_path", "") or "").strip()
    except Exception:
        path = ""
    if not path:
        return False
    if _dict_loaded == path:
        return True

    import jieba

    p = Path(path)
    if not p.is_file():
        logger.warning(
            "法律术语词典缺失: %s —— 退回 jieba 默认词典（会切碎法律术语，伤 BM25 召回）。"
            "请先运行 python scripts/build_legal_dict.py",
            path,
        )
        _dict_loaded = path  # 标记已尝试，避免每条文本都告警
        return False
    try:
        jieba.load_userdict(str(p))
        _dict_loaded = path
        logger.info("已加载法律术语词典: %s", path)
        return True
    except Exception:
        logger.exception("加载法律术语词典失败: %s", path)
        _dict_loaded = path
        return False


def _tokenize_zh(text: str) -> list[str]:
    import jieba

    _ensure_legal_dict()
    text = text.lower()
    raw = list(jieba.cut(text, cut_all=False))
    out: list[str] = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        out.append(t)
        for m in re.findall(r"[a-z0-9]+", t, flags=re.I):
            if len(m) > 1:
                out.append(m)
    return out if out else [" "]


def _bm25_cache_path(corpus: Path) -> Path:
    """缓存文件 = <语料目录>/.bm25_cache/<stem>.<指纹16位>.pkl。

    指纹绑定 缓存版本 + 语料绝对路径 + mtime_ns + size，语料一变即自动失效。
    """
    st = corpus.stat()
    key = (
        f"{_BM25_CACHE_VERSION}|{corpus.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return corpus.parent / ".bm25_cache" / f"{corpus.stem}.{digest}.pkl"


def _load_bm25_cache(cache_file: Path) -> dict | None:
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict) or obj.get("version") != _BM25_CACHE_VERSION:
            logger.info("BM25 缓存版本不符，重建: %s", cache_file)
            return None
        logger.info(
            "BM25 缓存命中: %s (%.0f MB)", cache_file,
            cache_file.stat().st_size / 1048576,
        )
        return obj
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 缓存损坏，将重建索引: %s (%s)", cache_file, exc)
        return None


def _save_bm25_cache(cache_file: Path, *, bm25: BM25Okapi, ids: list[str], meta: dict) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(
                {
                    "version": _BM25_CACHE_VERSION,
                    "bm25": bm25,
                    "ids": ids,
                    "meta": meta,
                },
                f,
                protocol=4,
            )
        os.replace(tmp, cache_file)
        # 清理同语料旧指纹的过期缓存
        stem_prefix = cache_file.name.split(".")[0]
        for old in cache_file.parent.glob(f"{stem_prefix}.*.pkl"):
            if old != cache_file:
                old.unlink(missing_ok=True)
        logger.info(
            "BM25 缓存已写入: %s (%.0f MB)", cache_file,
            cache_file.stat().st_size / 1048576,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 缓存写入失败（不影响本次使用，下次重建）: %s", exc)


def _load_corpus_disk(corpus_path: str) -> tuple[BM25Okapi, list[str], dict[str, dict]]:
    p = Path(corpus_path)
    if not p.is_file():
        raise FileNotFoundError(f"BM25 语料不存在: {p}（请先 python scripts/reindex.py）")

    cache_file = _bm25_cache_path(p)
    cached = _load_bm25_cache(cache_file)
    if cached is not None:
        return cached["bm25"], cached["ids"], cached["meta"]

    # ── 冷启动全量构建（分钟级），带进度日志 ──
    t0 = time.monotonic()
    logger.info("BM25 冷启动：开始分词 %s（分钟级，构建完成后写持久缓存 %s）", p, cache_file)
    ids: list[str] = []
    tokenized_corpus: list[list[str]] = []
    meta: dict[str, dict] = {}
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        nid = str(obj["id"])
        text = str(obj.get("text") or "")
        ids.append(nid)
        tokenized_corpus.append(_tokenize_zh(text))
        meta[nid] = dict(obj.get("metadata") or {})
        meta[nid]["_bm25_text"] = text
        n += 1
        if n % 5000 == 0:
            logger.info("BM25 分词进度: %d 条 (%.0fs)", n, time.monotonic() - t0)
    if not tokenized_corpus:
        return BM25Okapi([["empty"]]), [], {}
    logger.info(
        "BM25 分词完成: %d 条, %.0fs；开始构建 BM25Okapi", n, time.monotonic() - t0
    )
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 索引构建完成: %.0fs", time.monotonic() - t0)
    _save_bm25_cache(cache_file, bm25=bm25, ids=ids, meta=meta)
    return bm25, ids, meta


def _get_bm25(settings: Settings, *, corpus_path: str | None = None) -> tuple[BM25Okapi, list[str], dict[str, dict]]:
    """获取 BM25 索引。支持指定 corpus_path 用于中文/英文分离。线程安全。"""
    global _bm25, _bm25_ids, _bm25_meta
    global _bm25_cn, _bm25_ids_cn, _bm25_meta_cn

    with _bm25_lock:
        # 判断是否使用中文语料
        cn_path = getattr(settings, "bm25_corpus_path_cn", "data/bm25_cn_corpus.jsonl")
        if corpus_path:
            path = str(Path(corpus_path).resolve())
        else:
            path = str(Path(settings.bm25_corpus_path).resolve())

        # 中文路径使用独立缓存
        if cn_path in path or "_cn_" in path:
            if _bm25_cn is not None and _bm25_ids_cn is not None and _bm25_meta_cn is not None:
                return _bm25_cn, _bm25_ids_cn, _bm25_meta_cn
            _bm25_cn, _bm25_ids_cn, _bm25_meta_cn = _load_corpus_disk(path)
            logger.info("BM25(CN) 已加载: %s 条文档", len(_bm25_ids_cn))
            return _bm25_cn, _bm25_ids_cn, _bm25_meta_cn

        # 英文路径使用原缓存
        if _bm25 is not None and _bm25_ids is not None and _bm25_meta is not None:
            return _bm25, _bm25_ids, _bm25_meta
        _bm25, _bm25_ids, _bm25_meta = _load_corpus_disk(path)
        logger.info("BM25 已加载: %s 条文档", len(_bm25_ids))
        return _bm25, _bm25_ids, _bm25_meta


def bm25_search(
    settings: Settings,
    query: str,
    top_k: int,
    *,
    allowed_ids: frozenset[str] | None = None,
    corpus_path: str | None = None,
) -> list[tuple[str, float]]:
    """返回 (node_id, bm25_raw_score) 降序。支持指定 corpus_path。"""
    if top_k < 1:
        return []
    try:
        bm25, ids, _ = _get_bm25(settings, corpus_path=corpus_path)
    except FileNotFoundError:
        logger.warning("BM25 语料缺失，跳过 BM25 分支")
        return []
    if not ids:
        return []
    q_tokens = _tokenize_zh(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    if allowed_ids is not None:
        ranked = sorted(
            (i for i in range(len(ids)) if ids[i] in allowed_ids),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:top_k]
    else:
        ranked = sorted(
            range(len(ids)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]
    return [(ids[i], float(scores[i])) for i in ranked]


def node_with_score_from_bm25(
    node_id: str, bm25_score: float, meta_lookup: dict[str, dict]
) -> NodeWithScore | None:
    m = meta_lookup.get(node_id)
    if not m:
        return None
    text = str(m.get("_bm25_text") or "")
    meta = {k: v for k, v in m.items() if k != "_bm25_text"}
    node = TextNode(text=text, metadata=meta, id_=node_id)
    return NodeWithScore(node=node, score=bm25_score)
