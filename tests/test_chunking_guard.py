"""Bounded-behavior guards for src.chunking._recursive_split_text.

针对旧实现的三个病灶做回归防护：
1. rfind 从全文位置 0 回看 → O(n²) 且跨块回看；
2. 绝对位置门限随 start 增大行为改变 → 超长文本碎块爆炸（76K 字 → 1563 块）；
3. 合并分支字符串累积 → MemoryError。

本文件断言重写后的有界性（a/b/c 三条）对任意输入成立。
"""
from __future__ import annotations

import time

from src.chunking import _recursive_split_text

MAX_CHARS = 1024
OVERLAP = 128


def _dense_chinese(total_chars: int) -> str:
    """句号密集的中文文本（模拟裁判文书说理段）。"""
    sentence = "本院认为，被告人以非法占有为目的，秘密窃取他人财物，数额较大，其行为已构成盗窃罪。"
    reps = total_chars // len(sentence) + 1
    return (sentence * reps)[:total_chars]


def _assert_structure(chunks: list[dict], max_chars: int) -> None:
    assert chunks, "应至少产出一个块"
    for c in chunks:
        assert set(c.keys()) == {"text", "overlap_prev", "overlap_next"}
        assert isinstance(c["text"], str) and c["text"]
        assert len(c["text"]) <= max_chars, f"块长 {len(c['text'])} 超过 max_chars={max_chars}"


def test_dense_chinese_76k_chunk_count_and_size_bounded():
    text = _dense_chinese(76_000)
    chunks = _recursive_split_text(text, MAX_CHARS, OVERLAP)
    _assert_structure(chunks, MAX_CHARS)
    cap = len(text) // max(1, MAX_CHARS - OVERLAP) + 16
    assert len(chunks) <= cap, f"块数 {len(chunks)} 超过硬上限 {cap}"
    # 相邻块重叠语义：后块的 overlap_prev 取自前块文本尾部
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]["text"]
        expected = prev[-OVERLAP:] if len(prev) > OVERLAP else prev
        assert chunks[i]["overlap_prev"] == expected


def test_no_punctuation_500k_fast_and_bounded():
    text = "测" * 500_000  # 无任何标点/换行 → 每轮都在 end_candidate 硬切
    t0 = time.monotonic()
    chunks = _recursive_split_text(text, MAX_CHARS, OVERLAP)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"500K 无标点文本切分耗时 {elapsed:.2f}s，超过 5s 上限"
    _assert_structure(chunks, MAX_CHARS)
    cap = len(text) // max(1, MAX_CHARS - OVERLAP) + 16
    assert len(chunks) <= cap, f"块数 {len(chunks)} 超过硬上限 {cap}"
    # 合并后的小尾巴也不得超过 2×max_chars
    assert all(len(c["text"]) <= 2 * MAX_CHARS for c in chunks)


def test_empty_and_whitespace_return_empty():
    assert _recursive_split_text("", MAX_CHARS, OVERLAP) == []
    assert _recursive_split_text("   \n\t  \n", MAX_CHARS, OVERLAP) == []
    assert _recursive_split_text("\u3000" * 100, MAX_CHARS, OVERLAP) == []
    assert _recursive_split_text("", 10, 0) == []


def test_small_tail_merged_into_previous_within_cap():
    max_chars, overlap = 200, 20
    step = max_chars - overlap  # 180
    tail_len = 30  # < min_chunk_chars=50 → 必须并入前一块
    text = "汉" * (2 * step + tail_len)  # 390 字，无标点 → 硬切
    chunks = _recursive_split_text(text, max_chars, overlap)
    assert len(chunks) == 2, f"应为主块+合并块共 2 块，实际 {len(chunks)}"
    for c in chunks:
        assert set(c.keys()) == {"text", "overlap_prev", "overlap_next"}
        assert c["text"]
        # 任何块（含合并块）都不得超过 2×max_chars
        assert len(c["text"]) <= 2 * max_chars
    # 常规块 ≤ max_chars；合并块允许超过 max_chars（并入尾巴所致）
    assert len(chunks[0]["text"]) <= max_chars
    merged = chunks[1]["text"]
    assert len(merged) > max_chars, "尾巴未并入前一块（合并块应超过单块上限）"
    assert merged.endswith("汉" * tail_len), "尾巴内容丢失"


def test_total_output_chars_bounded():
    # 总字符量（text + overlap_prev + overlap_next，overlap 重复计入）
    # ≤ len(text) × 1.6 + 4096
    for text in (
        _dense_chinese(76_000),
        "测" * 500_000,
        _dense_chinese(76_000) + "无标点尾巴" * 7,
    ):
        chunks = _recursive_split_text(text, MAX_CHARS, OVERLAP)
        total = sum(
            len(c["text"]) + len(c["overlap_prev"]) + len(c["overlap_next"])
            for c in chunks
        )
        budget = len(text) * 1.6 + 4096
        assert total <= budget, f"总字符 {total} 超预算 {budget}（len={len(text)}）"
