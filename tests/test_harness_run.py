"""Tests for Agent Harness /agent/run endpoint and JSON parsing (M11 fix)."""
from __future__ import annotations


class TestJsonExtraction:
    """Verify robust JSON extraction from LLM output."""

    def test_extract_json_object_from_fenced_block(self):
        from src.agent.harness import _extract_json_object
        text = 'Here is the result:\n```json\n{"passed": true, "issues": []}\n```'
        result = _extract_json_object(text)
        assert result is not None
        assert result["passed"] is True

    def test_extract_json_object_from_plain_text(self):
        from src.agent.harness import _extract_json_object
        text = 'The evaluation is {"grounded": true, "safe": true, "unsupported_claims": []} as shown.'
        result = _extract_json_object(text)
        assert result is not None
        assert result["grounded"] is True

    def test_extract_json_object_returns_none_for_invalid(self):
        from src.agent.harness import _extract_json_object
        text = "No JSON here, just plain text."
        result = _extract_json_object(text)
        assert result is None

    def test_extract_json_array_from_fenced_block(self):
        from src.agent.harness import _extract_json_array
        text = '```json\n[{"step": 1, "action": "order_lookup"}]\n```'
        result = _extract_json_array(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["step"] == 1

    def test_extract_json_array_from_plain_text(self):
        from src.agent.harness import _extract_json_array
        text = 'Plan: [{"step": 1, "action": "check"}, {"step": 2, "action": "reply"}]'
        result = _extract_json_array(text)
        assert result is not None
        assert len(result) == 2

    def test_extract_json_array_returns_none_for_invalid(self):
        from src.agent.harness import _extract_json_array
        text = "No array here."
        result = _extract_json_array(text)
        assert result is None

    def test_extract_json_handles_nested_objects(self):
        """Greedy regex would match too much — outermost braces should work."""
        from src.agent.harness import _extract_json_object
        text = '{"outer": {"inner": "value"}, "list": [1, 2]}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["outer"]["inner"] == "value"
