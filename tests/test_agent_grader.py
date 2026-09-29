"""Agent grader test."""
from src.agent_grader import ngram_overlap

def test_ngram_overlap():
    overlap = ngram_overlap("如何重置密码", "如何重置登录密码？")
    assert overlap > 0.3
    assert overlap < 1.0

def test_ngram_overlap_empty():
    assert ngram_overlap("", "登录") == 0.0
