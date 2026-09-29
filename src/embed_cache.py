"""Embedding 磁盘缓存（断点续跑）+ 建库进度落盘。

为什么需要这个模块
------------------
8.8 万节点在 8GB 显存的机器上编码要以小时计，而原实现把所有向量攒在内存里、
**只在最后一步才写盘**。后果是：编码跑完之后任何一步失败（Qdrant 建库、
BM25 语料落盘、集合维度核对）都要**重烧一整轮**——这正是"等半天啥都没搞出来"
的结构性原因。

这里做两件事：
1. **向量按内容寻址追加落盘**（key = 待编码文本的 md5）。任何一步崩了，重跑时
   已算过的直接命中，不重新编码。切块策略变化时，未变的片段同样能复用。
2. **进度实时落盘**（`progress.json`）。done / total / 速率 / ETA / 缓存命中数，
   外部不用盯终端就能判断"还剩多久"和"是不是卡住了"。

缓存失效判据（宁可重算，不可用错向量）
--------------------------------------
缓存目录以 `模型名 + 维度 + 精度 + 版本` 为标签分目录。换模型、改维度或改精度
→ 标签变 → 旧缓存自动视为不存在，而不是读出语义不同的向量。

崩溃一致性
----------
一行向量对应一行 key：**先追加 vectors，再追加 keys**。进程在两步之间被杀，
磁盘上会出现"向量行比 key 行多"的孤儿尾巴；加载时按 `min(向量行, key 行)`
对齐，孤儿行被忽略（索引映射仍然正确，只是浪费几 KB 空间），不会串位。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DTYPE = np.float16
_FILE_VECTORS = "vectors.raw"
_FILE_KEYS = "keys.txt"
_FILE_META = "meta.json"
# 缓存语义版本：向量的"含义"变了就 +1（例如 normalize 口径、模型池变更）
CACHE_VERSION = "v1"


def text_key(text: str) -> str:
    """待编码文本 → 缓存键。用全文 md5，不用前缀（前缀会误判，见 HANDOFF §9.4）。"""
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def cache_tag(model_name: str, dim: int, fp16: bool) -> str:
    slug = model_name.rstrip("/").split("/")[-1].replace(" ", "_")
    return f"{slug}_d{dim}_{'fp16' if fp16 else 'fp32'}_{CACHE_VERSION}"


class EmbeddingCache:
    """内容寻址的向量缓存（追加式，单进程写）。

    dtype 跟精度走：fp16 跑用 float16 存（181MB / 8.8 万点），fp32 跑用 float32 存。
    不能图省事统一 fp16 —— 那会让 CPU/fp32 路线上"命中缓存的向量"与"当场编码的
    向量"精度不同，相似度在阈值附近的题就会飘（评测不可复现）。
    """

    def __init__(self, root: Path, *, dim: int, tag: str, dtype=np.float16) -> None:
        self.dim = int(dim)
        self.tag = tag
        self.dtype = np.dtype(dtype)
        self.dir = Path(root) / tag
        self.vec_path = self.dir / _FILE_VECTORS
        self.key_path = self.dir / _FILE_KEYS
        self.meta_path = self.dir / _FILE_META

        self._index: dict[str, int] = {}
        self._rows: np.ndarray | None = None  # memmap (n, dim)，只装已落盘的老数据
        self._base_n = 0            # memmap 里的行数；行号 >= _base_n 的走 _added
        self._added: list[np.ndarray] = []  # 本次进程新写入的行（未重载 memmap，靠它命中）
        self._pending_vec: list[np.ndarray] = []
        self._pending_key: list[str] = []

        self.hits = 0
        self.stored = 0
        self.loaded = 0
        self.orphans = 0

    # ── 加载 ──
    def load(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.meta_path.is_file():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            if (
                int(meta.get("dim", -1)) != self.dim
                or meta.get("tag") != self.tag
                or meta.get("dtype") != self.dtype.name
            ):
                logger.warning(
                    "缓存目录标签不匹配（dim/tag 变了），忽略旧缓存: %s", self.dir
                )
                return
        elif self.vec_path.is_file() or self.key_path.is_file():
            logger.warning("缓存缺少 meta.json，忽略旧缓存: %s", self.dir)
            return

        if not self.key_path.is_file() or not self.vec_path.is_file():
            return

        keys = [
            line.strip()
            for line in self.key_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vec_rows = self.vec_path.stat().st_size // (self.dim * self.dtype.itemsize)
        self.orphans = max(0, vec_rows - len(keys))
        self._n = min(vec_rows, len(keys))
        if self._n == 0:
            return
        keys = keys[: self._n]
        self._rows = np.memmap(
            self.vec_path, dtype=self.dtype, mode="r", shape=(self._n, self.dim)
        )
        self._base_n = self._n
        self._index = {k: i for i, k in enumerate(keys)}
        # 文本完全相同但被切到不同位置 → 一个 key 一行即可，重复 key 保留首次出现
        self.loaded = len(keys)
        if self.orphans:
            logger.warning(
                "缓存有 %s 个孤儿向量行（上次中断留下的尾巴），已忽略", self.orphans
            )
        logger.info("Embedding 缓存加载: %s 条可用 @ %s", len(self._index), self.dir)

    # ── 读 ──
    def get(self, key: str) -> list[float] | None:
        i = self._index.get(key)
        if i is None:
            return None
        if i < self._base_n:
            # 旧数据：从磁盘 memmap 读（本次进程没编码过它）
            if self._rows is None:
                return None
            vec = self._rows[i]
        else:
            # 本进程刚算出来的：必须能立刻命中。否则同一轮里的重复文本会被重编
            # ——实测语料有 10,956 条跨文档同文本，漏了就是白烧 14% 的 GPU 时间。
            off = i - self._base_n
            if off >= len(self._added):
                return None
            vec = self._added[off]
        self.hits += 1
        return np.asarray(vec, dtype=np.float32).tolist()

    # ── 写（缓冲）──
    def put(self, key: str, vec: Any) -> None:
        if key in self._index:
            return
        arr = np.asarray(vec, dtype=self.dtype)
        if arr.size != self.dim:
            raise ValueError(f"向量维度 {arr.size} != 缓存维度 {self.dim}")
        self._index[key] = self._base_n + len(self._added)
        self._added.append(arr)
        self._pending_key.append(key)
        self._pending_vec.append(arr)

    def flush(self, *, sync: bool = False) -> None:
        """追加落盘。顺序固定：先 vectors 后 keys（见模块 docstring 的崩溃一致性）。"""
        if not self._pending_key:
            return
        rows = np.stack(self._pending_vec).astype(self.dtype, copy=False)
        with open(self.vec_path, "ab") as f:
            f.write(rows.tobytes())
            f.flush()
            if sync:
                os.fsync(f.fileno())
        with open(self.key_path, "a", encoding="utf-8") as f:
            f.write("".join(k + "\n" for k in self._pending_key))
            f.flush()
            if sync:
                os.fsync(f.fileno())
        self.stored += len(self._pending_key)
        self._n = self._base_n + len(self._added)
        self._pending_vec.clear()
        self._pending_key.clear()
        if self.meta_path.is_file() is False:
            self.meta_path.write_text(
                json.dumps(
                    {"tag": self.tag, "dim": self.dim, "dtype": self.dtype.name},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def reset(self) -> None:
        """--fresh-cache：作废现有缓存（改名 .stale-<ts>，不静默删除）。"""
        if not self.dir.is_dir():
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stale = self.dir.with_name(f"{self.dir.name}.stale-{stamp}")
        try:
            self.dir.rename(stale)
            logger.warning("Embedding 缓存已作废并移至: %s", stale)
        except OSError as e:  # 目录被别的进程占用（例如上一次建库还没退出）
            raise RuntimeError(
                f"无法移出旧缓存 {self.dir}：{e}。"
                "检查是否有另一个 build_index / 检索进程正在运行（本地 Qdrant 与缓存都不支持并发打开）。"
            ) from e
        self._index, self._rows, self._n = {}, None, 0
        self._base_n, self._added, self._pending_vec, self._pending_key = 0, [], [], []


class Progress:
    """建库进度落盘 + 速率/ETA 估算（滑动窗口，不被开头几百个节点拉偏）。"""

    def __init__(self, path: Path, *, total: int, meta: dict[str, Any]) -> None:
        self.path = Path(path)
        self.total = int(total)
        self.meta = meta
        self.done = 0
        self.t0 = time.perf_counter()
        self._win_t = self.t0
        self._win_done = 0
        self._rate = 0.0
        self.phase = "embedding"

    def update(self, done: int, *, phase: str | None = None) -> None:
        self.done = int(done)
        now = time.perf_counter()
        if phase:
            self.phase = phase
        # 滑动窗口速率：约每 60s 重设一次基线，保留此前 EMA
        if now - self._win_t >= 60.0 or self._win_done == 0:
            inst = (self.done - self._win_done) / max(now - self._win_t, 1e-6)
            self._rate = inst if self._rate == 0 else 0.5 * self._rate + 0.5 * inst
            self._win_t, self._win_done = now, self.done
        self.write()

    def write(self) -> None:
        elapsed = time.perf_counter() - self.t0
        remaining_nodes = max(self.total - self.done, 0)
        eta = remaining_nodes / self._rate if self._rate > 0.05 else None
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phase": self.phase,
            "done": self.done,
            "total": self.total,
            "percent": round(100.0 * self.done / max(self.total, 1), 2),
            "elapsed_seconds": round(elapsed, 1),
            "rate_nodes_per_second": round(self._rate, 2),
            "eta_seconds": round(eta, 0) if eta is not None else None,
            "eta_at": time.strftime(
                "%H:%M:%S", time.localtime(time.time() + eta)
            )
            if eta is not None
            else None,
            "pid": os.getpid(),
            **self.meta,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
