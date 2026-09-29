"""Embedding 断点缓存与进度落盘测试（D2）。

背景：建库要在 8GB 显存机器上编码 8.8 万节点（小时级），而原实现把向量全攒在内存、
**只在最后一步写盘** → 编码跑完后任何一步失败（Qdrant 建库 / BM25 落盘 / 维度核对）
都要重烧一整轮。新加的 `src/embed_cache.py` 把向量按内容寻址追加落盘，并把进度
实时写 `progress.json`。

本文件锁四条，都是"新代码自己会不会引入静默错误"：
1. **往返一致**：写入 → 重载 → 命中，向量不能串位（串位 = 检索悄悄变错且不报错）；
2. **崩溃一致性**：进程在"向量已写、key 未写"之间被杀，重载后只能浪费空间，
   绝不能让文本 A 拿到文本 B 的向量；
3. **失效判据**：维度 / 精度 / 模型标签变了 → 旧缓存整体作废，不复用语义不同的向量；
4. **进度可观测**：ETA / 速率 / 阶段外部可读（这是"还要多久"的非终端来源）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.embed_cache import EmbeddingCache, Progress, cache_tag, text_key

DIM = 8


def _vec(i: int) -> list[float]:
    """每个下标产生一把可区分的向量（串位立刻能被断言抓到）。"""
    return [float(i) + 0.5] * DIM


def _cache(tmp_path, *, dim=DIM, fp16=True) -> EmbeddingCache:
    tag = cache_tag("BAAI/bge-m3", dim, fp16)
    return EmbeddingCache(
        tmp_path,
        dim=dim,
        tag=tag,
        dtype=np.float16 if fp16 else np.float32,
    )


class TestRoundtrip:
    def test_put_flush_reload_hit(self, tmp_path):
        c = _cache(tmp_path, fp16=False)
        c.load()
        for i in range(5):
            c.put(text_key(f"文本{i}"), _vec(i))
        c.flush()
        assert c.stored == 5

        c2 = _cache(tmp_path, fp16=False)
        c2.load()
        assert c2.loaded == 5
        for i in range(5):
            got = c2.get(text_key(f"文本{i}"))
            assert got is not None, f"文本{i} 应命中"
            assert got == pytest.approx(_vec(i), abs=1e-6), "向量串位"

    def test_fp16_roundtrip_within_tolerance(self, tmp_path):
        c = _cache(tmp_path, fp16=True)
        c.load()
        c.put(text_key("a"), _vec(1))
        c.flush()

        c2 = _cache(tmp_path, fp16=True)
        c2.load()
        assert c2.get(text_key("a")) == pytest.approx(_vec(1), abs=1e-3)

    def test_same_text_stored_once(self, tmp_path):
        """同一文本被切到不同位置（跨文档重复，语料里预期有 1 万+）只落一行。"""
        c = _cache(tmp_path, fp16=False)
        c.load()
        c.put(text_key("dup"), _vec(7))
        c.put(text_key("dup"), _vec(7))
        c.flush()
        assert c.stored == 1

    def test_miss_returns_none(self, tmp_path):
        c = _cache(tmp_path, fp16=False)
        c.load()
        assert c.get(text_key("没见过")) is None

    def test_intra_run_hit_without_reload(self, tmp_path):
        """本进程刚 put+flush 的向量必须**不重新 load() 就能命中**。

        真踩过的坑：首版实现只从 memmap 读，而 memmap 是 load() 时按旧行数开的，
        于是同一轮里 `hits` 恒为 0 —— 实测语料有 10,956 条跨文档同文本，
        等于白烧 14% 的 GPU 时间。判据用 get() 拿回自己的向量，不看 hits 计数。
        """
        c = _cache(tmp_path, fp16=False)
        c.load()
        for i in range(4):
            c.put(text_key(f"a{i}"), _vec(i))
        c.flush()
        for i in range(4):
            got = c.get(text_key(f"a{i}"))
            assert got is not None, f"第{i}条已写入却没命中（同轮重复文本会重编）"
            assert got == pytest.approx(_vec(i), abs=1e-6), "同轮命中却取回错向量"

        # 未 flush 的 pending 同样要能命中（同一 batch 内就重复的情况）
        c.put(text_key("pending"), _vec(9))
        assert c.get(text_key("pending")) == pytest.approx(_vec(9), abs=1e-6)

    def test_two_flush_batches_stay_aligned_after_reload(self, tmp_path):
        """分两批 flush（模拟逐 batch 落盘）后重载：行号必须仍然对得上。"""
        c = _cache(tmp_path, fp16=False)
        c.load()
        for i in range(3):
            c.put(text_key(f"first{i}"), _vec(i))
        c.flush()
        for i in range(3, 6):
            c.put(text_key(f"second{i}"), _vec(i))
        c.flush()

        c2 = _cache(tmp_path, fp16=False)
        c2.load()
        assert c2.loaded == 6
        for i in range(3):
            assert c2.get(text_key(f"first{i}")) == pytest.approx(_vec(i), abs=1e-6)
            assert c2.get(text_key(f"second{i + 3}")) == pytest.approx(_vec(i + 3), abs=1e-6)


class TestCrashConsistency:
    def test_orphan_vector_rows_are_ignored_not_misaligned(self, tmp_path):
        """模拟"写完向量、还没写 key 就被杀"：尾部多出的向量行必须被忽略。

        判据不是"文件变小"，而是**已登记的 key 仍取回自己的向量**——
        一旦按向量行数对齐就会串位，那是最危险的静默错误。
        """
        c = _cache(tmp_path, fp16=False)
        c.load()
        for i in range(3):
            c.put(text_key(f"t{i}"), _vec(i))
        c.flush()

        # 手工追加一行"孤儿向量"（没有对应 key）
        with open(c.vec_path, "ab") as f:
            f.write(np.asarray(_vec(99), dtype=np.float32).tobytes())

        c2 = _cache(tmp_path, fp16=False)
        c2.load()
        assert c2.orphans == 1
        assert c2.loaded == 3
        for i in range(3):
            assert c2.get(text_key(f"t{i}")) == pytest.approx(_vec(i), abs=1e-6)

    def test_pending_flush_survives_second_batch(self, tmp_path):
        c = _cache(tmp_path, fp16=False)
        c.load()
        c.put(text_key("batch1"), _vec(1))
        c.flush()
        c.put(text_key("batch2"), _vec(2))
        c.flush()

        c2 = _cache(tmp_path, fp16=False)
        c2.load()
        assert c2.get(text_key("batch1")) == pytest.approx(_vec(1), abs=1e-6)
        assert c2.get(text_key("batch2")) == pytest.approx(_vec(2), abs=1e-6)


class TestInvalidation:
    def test_different_tag_ignores_old_cache(self, tmp_path):
        """换模型 / 改维度 / 改精度 → 标签变 → 旧向量语义不同，必须整体作废。"""
        c = EmbeddingCache(tmp_path, dim=DIM, tag="bge-m3_d8_fp16_v1", dtype=np.float16)
        c.load()
        c.put(text_key("x"), _vec(3))
        c.flush()

        other = EmbeddingCache(
            tmp_path, dim=DIM, tag="bge-large_d8_fp16_v1", dtype=np.float16
        )
        other.load()
        assert other.get(text_key("x")) is None

    def test_dtype_change_ignores_old_cache(self, tmp_path):
        """同标签目录下 meta.json 记的 dtype 与本次不符 → 忽略（不静默按错 dtype 解释）。"""
        c = _cache(tmp_path, fp16=True)
        c.load()
        c.put(text_key("x"), _vec(4))
        c.flush()
        # 手动把 meta 的 tag 留同、dtype 改成不匹配的另一侧实例
        mismatched = EmbeddingCache(tmp_path, dim=DIM, tag=c.tag, dtype=np.float32)
        mismatched.load()
        assert mismatched.get(text_key("x")) is None

    def test_dim_change_ignores_old_cache(self, tmp_path):
        c = _cache(tmp_path, fp16=False)
        c.load()
        c.put(text_key("x"), _vec(5))
        c.flush()

        narrow = EmbeddingCache(
            tmp_path, dim=DIM + 4, tag=f"bge-m3_d{DIM + 4}_fp32_v1", dtype=np.float32
        )
        narrow.load()
        assert narrow.get(text_key("x")) is None

    def test_reset_moves_dir_aside(self, tmp_path):
        c = _cache(tmp_path, fp16=False)
        c.load()
        c.put(text_key("x"), _vec(6))
        c.flush()
        old_dir = c.dir

        c.reset()
        assert not old_dir.is_dir()
        assert list(tmp_path.glob(f"{old_dir.name}.stale-*"))
        assert c.get(text_key("x")) is None

    def test_put_wrong_dimension_raises(self, tmp_path):
        c = _cache(tmp_path, fp16=False)
        c.load()
        with pytest.raises(ValueError):
            c.put(text_key("bad"), [1.0, 2.0])


class TestProgress:
    def test_progress_json_has_eta_and_phase(self, tmp_path):
        p = tmp_path / "progress.json"
        prog = Progress(p, total=1000, meta={"model": "BAAI/bge-m3", "dim": DIM})
        prog.update(250)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["done"] == 250
        assert data["total"] == 1000
        assert data["percent"] == 25.0
        assert data["rate_nodes_per_second"] > 0
        assert data["eta_seconds"] is not None
        assert data["phase"] == "embedding"
        assert data["model"] == "BAAI/bge-m3"

        prog.phase = "qdrant_upsert"
        prog.update(1000)
        data2 = json.loads(p.read_text(encoding="utf-8"))
        assert data2["phase"] == "qdrant_upsert"
        assert data2["eta_seconds"] == 0

    def test_progress_write_is_atomic(self, tmp_path):
        """进度文件用 tmp + replace：外部读到的永远是完整 JSON，不会读到半截。"""
        p = tmp_path / "progress.json"
        prog = Progress(p, total=10, meta={})
        for i in range(1, 11):
            prog.update(i)
            json.loads(p.read_text(encoding="utf-8"))  # 每次都能解析
        assert not list(tmp_path.glob("*.tmp"))
