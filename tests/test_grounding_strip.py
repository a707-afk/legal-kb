"""Grounding strip test."""
from src.citation import strip_unsupported_sentences, verify_grounding


def test_strip_unsupported_empty():
    assert strip_unsupported_sentences("", "context") == ""
    assert strip_unsupported_sentences("text", "") == "text"


def test_verify_grounding_all_supported():
    answer = "退货政策允许7天无理由退货。未开封商品可全额退款。"
    evidence = "我们的退货政策允许7天无理由退货。未开封商品可以享受全额退款。"
    result = verify_grounding(answer, evidence)
    assert isinstance(result, dict)
    assert "unsupported_sentences" in result
    assert len(result["unsupported_sentences"]) == 0


def test_verify_grounding_partially_unsupported():
    answer = "退货政策允许7天无理由退货。地球是太阳系第三颗行星。"
    evidence = "我们的退货政策允许7天无理由退货。未开封商品可以享受全额退款。"
    result = verify_grounding(answer, evidence)
    assert isinstance(result["unsupported_sentences"], list)
    assert len(result["unsupported_sentences"]) >= 1


def test_verify_grounding_empty_inputs():
    result = verify_grounding("", "some evidence")
    assert result["unsupported_sentences"] == []
    assert result["passed"] is True

    result = verify_grounding("some answer", "")
    assert result["unsupported_sentences"] == []
    assert result["passed"] is True


def test_strip_unsupported_removes_bad_sentences():
    text = "退货政策允许7天无理由退货。地球是太阳系第三颗行星。"
    context = "我们的退货政策允许7天无理由退货。未开封商品可以享受全额退款。"
    stripped = strip_unsupported_sentences(text, context)
    assert "退货政策" in stripped
