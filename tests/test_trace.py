"""检索链路 trace 测试（D-15）。

契约：一次查询一个 JSONL 文件、一行一个 stage、可按 query_id O(1) 回放；
**只落候选摘要，不落全文**（避免泄露敏感信息）。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    # traces_root 每次都读环境变量，无需清缓存
    yield tmp_path / "traces"


class TestTraceRecorder:
    def test_record_and_load_roundtrip(self, trace_dir):
        from src.trace import TraceRecorder, load_trace

        tr = TraceRecorder("q-test-001")
        tr.record("route", language="zh")
        tr.record("parallel", vector=20, bm25=18)
        tr.finish(final_count=5)

        rows = load_trace("q-test-001")
        assert [r["stage"] for r in rows] == ["route", "parallel", "finish"]
        assert rows[0]["language"] == "zh"
        assert rows[-1]["final_count"] == 5
        assert all(r["query_id"] == "q-test-001" for r in rows)
        assert all("elapsed_ms" in r for r in rows)

    def test_disabled_is_noop(self, trace_dir):
        from src.trace import TraceRecorder, load_trace

        tr = TraceRecorder("q-disabled", enabled=False)
        tr.record("route", language="zh")
        tr.finish(final_count=0)
        assert load_trace("q-disabled") == []
        assert not list(trace_dir.rglob("*.jsonl"))

    def test_one_file_per_query(self, trace_dir):
        from src.trace import TraceRecorder, trace_path

        a = TraceRecorder("q-a")
        b = TraceRecorder("q-b")
        a.record("route")
        b.record("route")
        assert trace_path("q-a") != trace_path("q-b")
        assert trace_path("q-a").is_file()
        assert trace_path("q-b").is_file()

    def test_candidate_brief_has_no_full_text(self, trace_dir):
        """只落 file / score / status_code / heading_path，不落全文。"""
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.trace import TraceRecorder, load_trace

        node = TextNode(
            text="这是一段很长的法条全文" * 50,
            metadata={"file_path": "statute/x.md", "status_code": 3, "heading_path": "第一章 > 第二条"},
        )
        tr = TraceRecorder("q-brief")
        tr.stage_candidates("rerank", [NodeWithScore(node=node, score=0.87)])
        row = load_trace("q-brief")[0]
        assert row["count"] == 1
        assert row["top"][0]["score"] == 0.87
        assert row["top"][0]["status_code"] == 3
        assert "全文" not in json.dumps(row, ensure_ascii=False)

    def test_write_failure_does_not_raise(self, trace_dir, monkeypatch):
        """trace 写失败绝不能拖垮检索。"""
        from src.trace import TraceRecorder

        tr = TraceRecorder("q-fail")
        monkeypatch.setattr(Path, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        tr.record("route")  # 不应抛

    def test_unsafe_query_id_is_sanitized(self, trace_dir):
        from src.trace import TraceRecorder, trace_path

        tr = TraceRecorder("q/../evil:name")
        tr.record("route")
        p = trace_path("q/../evil:name")
        assert p.is_file()
        assert ".." not in p.name and "/" not in p.name


class TestCreateTrace:
    def test_create_trace_respects_setting(self, trace_dir):
        import src.trace as trace_mod

        with patch("src.config_LEGACY_REFERENCE.get_settings") as gs:
            gs.return_value = type("S", (), {"retrieval_trace_enabled": False})()
            tr = trace_mod.create_trace("q-off")
            assert tr.enabled is False

    def test_new_query_id_shape(self):
        from src.trace import new_query_id

        qid = new_query_id()
        assert qid.startswith("q-")
        assert len(qid.split("-")) == 4  # q / 日期 / 时间 / 随机
