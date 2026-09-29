"""citation_verify：字符 overlap 与句级 grounding。"""
from __future__ import annotations

from src.citation import (
    GroundingReport,
    SentenceGrounding,
    citation_overlap_ratio,
    sentence_level_grounding,
)


def _chunk(text: str, node_id: str = "c1") -> dict:
    return {"text": text, "node_id": node_id, "file_name": "a.md", "file_path": "a/a.md"}


def test_citation_overlap_ratio_hit():
    chunk = "标准退款流程：提交工单、审核、3-5 工作日原路退回。"
    # overlap 需 chunk 内 ≥12 字连续子串出现在答案中
    answer = f"请按以下办理：{chunk}"
    ratio = citation_overlap_ratio(answer, [chunk])
    assert ratio > 0.0


def test_citation_overlap_ratio_no_chunks_returns_one():
    assert citation_overlap_ratio("任意答案", []) == 1.0


def test_sentence_level_grounding_pass_when_draft_matches_chunk():
    draft = "建议按标准退款流程处理，引导客户提交工单。"
    chunks = [_chunk("标准退款流程：提交工单、审核、3-5 工作日原路退回。", "n-refund")]
    report = sentence_level_grounding(draft, chunks, prefer_embedding=False)
    assert report.passed is True
    assert report.method == "ngram"
    assert report.unsupported_sentence_rate == 0.0
    assert len(report.sentences) >= 1
    assert report.sentences[0].unsupported is False


def test_sentence_level_grounding_fail_when_hallucinated():
    draft = "退款将在 24 小时内自动到账，无需任何审批。"
    chunks = [_chunk("标准退款流程：提交工单、审核、3-5 工作日原路退回。")]
    report = sentence_level_grounding(draft, chunks, prefer_embedding=False)
    assert report.passed is False
    assert report.unsupported_sentence_rate > 0.0
    assert any(s.unsupported for s in report.sentences)


def test_sentence_level_grounding_no_chunks_all_unsupported():
    report = sentence_level_grounding("这是一句测试。", [], prefer_embedding=False)
    assert report.passed is False
    assert report.unsupported_sentence_rate == 1.0


def test_sentence_level_grounding_empty_answer():
    report = sentence_level_grounding("   ", [_chunk("证据")], prefer_embedding=False)
    assert report.passed is False
    assert report.feedback == "无有效句子"


def test_grounding_report_to_dict_shape():
    report = GroundingReport(
        sentences=[
            SentenceGrounding(
                sentence="测试句。",
                support_score=0.5,
                best_chunk_id="c1",
                unsupported=False,
                method="ngram",
            )
        ],
        overlap_ratio=0.8,
        unsupported_sentence_rate=0.0,
        passed=True,
        method="ngram",
        feedback="ok",
    )
    d = report.to_dict()
    assert d["passed"] is True
    assert d["method"] == "ngram"
    assert len(d["sentences"]) == 1
    assert d["sentences"][0]["best_chunk_id"] == "c1"
