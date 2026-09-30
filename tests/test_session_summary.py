"""P3：模型上下文窗口配置 + session_mgr 多轮摘要接入的回归测试。

覆盖三块：
1. 窗口配置推导出的字符预算与配置化前的硬编码值一致（8000 / 4000）
2. _format_evidence_block 的 total_chars 可覆盖（向后兼容）
3. session_mgr 注册表（record_turn / summarize_last_n / LRU 有界）与
   harness 的防御式摘要获取（_get_session_summary）
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_sessions():
    """每个用例前后清空会话注册表，避免跨用例串话。"""
    from src.session_mgr import clear_sessions
    clear_sessions()
    yield
    clear_sessions()


class TestContextWindowConfig:
    def test_default_generator_budget_matches_legacy_8000(self):
        from src.agent.harness import DEFAULT_EVIDENCE_TOTAL_CHARS
        assert DEFAULT_EVIDENCE_TOTAL_CHARS == 8000

    def test_default_evaluator_budget_matches_legacy_4000(self):
        from src.agent.harness import EVALUATOR_EVIDENCE_TOTAL_CHARS
        assert EVALUATOR_EVIDENCE_TOTAL_CHARS == 4000

    def test_budget_ratio_respects_hard_cap(self):
        """即使把比例调到 1.0，字符预算也不突破 EVIDENCE_BUDGET_RATIO 上限。"""
        from src.agent.harness import (
            EVIDENCE_BUDGET_RATIO,
            MODEL_CONTEXT_WINDOW,
            TOKEN_ESTIMATE_RATIO,
            _chars_budget,
        )
        cap_chars = int(MODEL_CONTEXT_WINDOW * EVIDENCE_BUDGET_RATIO / TOKEN_ESTIMATE_RATIO)
        assert _chars_budget(1.0) == cap_chars
        assert _chars_budget(EVIDENCE_BUDGET_RATIO) == cap_chars

    def test_config_values_sane(self):
        from src.agent import harness
        assert harness.MODEL_CONTEXT_WINDOW > 0
        assert harness.TOKEN_ESTIMATE_RATIO > 0
        assert 0 < harness.EVIDENCE_BUDGET_RATIO <= 1.0
        assert harness.HISTORY_SUMMARY_MAX_CHARS > 0


class TestEvidenceBlockBudget:
    def _obs(self, n_snippets: int = 3, text_len: int = 300) -> list[dict]:
        return [{
            "tool": "local_search",
            "result": {
                "snippets": [
                    {"text": f"第{i}条" + "x" * text_len, "title": "测试法",
                     "file_name": f"f{i}.md", "heading": f"第{i}条", "score": 0.9 - i * 0.1}
                    for i in range(n_snippets)
                ],
            },
        }]

    def test_default_total_chars_uses_config(self):
        from src.agent.harness import DEFAULT_EVIDENCE_TOTAL_CHARS, _format_evidence_block
        # 构造远超默认预算的观测：100 条 × 300 字
        block = _format_evidence_block(self._obs(n_snippets=100))
        # 截断标记出现，且块长度不超过预算 + 单条溢出余量
        assert "已截断" in block
        assert len(block) <= DEFAULT_EVIDENCE_TOTAL_CHARS + 400

    def test_explicit_total_chars_override(self):
        """显式传值仍然生效（向后兼容旧调用方）。"""
        from src.agent.harness import _format_evidence_block
        block = _format_evidence_block(self._obs(n_snippets=50), total_chars=500)
        assert "已截断" in block
        assert len(block) < 1200  # 500 预算 + 抬头/截断标记的余量

    def test_small_input_not_truncated(self):
        from src.agent.harness import _format_evidence_block
        block = _format_evidence_block(self._obs(n_snippets=2, text_len=50))
        assert "已截断" not in block
        assert "[1]" in block and "[2]" in block


class TestSessionRegistry:
    def test_record_turn_and_summarize(self):
        from src.session_mgr import get_session, record_turn
        record_turn("s1", user_message="劳动合同法第几条？", assistant_message="第四十六条……")
        mem = get_session("s1")
        assert mem is not None
        assert mem.turn_count() == 1
        summary = mem.summarize_last_n(3)
        assert "劳动合同法" in summary
        assert "第四十六条" in summary  # 助手侧也进摘要（追问需要上一答的落点）

    def test_get_session_missing_returns_none(self):
        from src.session_mgr import get_session
        assert get_session("nope") is None

    def test_record_turn_none_session_is_noop(self):
        from src.session_mgr import record_turn, session_count
        record_turn(None, user_message="q", assistant_message="a")
        assert session_count() == 0

    def test_summarize_empty_session_returns_empty(self):
        from src.session_mgr import get_or_create_session
        assert get_or_create_session("s2").summarize_last_n(3) == ""

    def test_messages_bounded(self):
        from src.session_mgr import MAX_MESSAGES_PER_SESSION, get_or_create_session
        mem = get_or_create_session("s3")
        for i in range(MAX_MESSAGES_PER_SESSION + 30):
            mem.add_message("user", f"m{i}")
        assert len(mem.messages) == MAX_MESSAGES_PER_SESSION

    def test_registry_lru_eviction(self):
        import src.session_mgr as sm
        for i in range(sm.MAX_SESSIONS + 10):
            sm.record_turn(f"sid{i}", user_message="q", assistant_message="a")
        assert sm.session_count() == sm.MAX_SESSIONS
        assert sm.get_session("sid0") is None  # 最旧的被淘汰


class TestHarnessSessionSummary:
    def test_no_session_id_returns_empty(self):
        from src.agent.harness import _get_session_summary
        assert _get_session_summary(None) == ""

    def test_unknown_session_returns_empty(self):
        from src.agent.harness import _get_session_summary
        assert _get_session_summary("never-seen") == ""

    def test_summary_injected_and_truncated(self):
        from src.agent.harness import HISTORY_SUMMARY_MAX_CHARS, _get_session_summary
        from src.session_mgr import record_turn
        record_turn("s4", user_message="长问题" * 300, assistant_message="长回答" * 300)
        summary = _get_session_summary("s4")
        assert summary
        assert len(summary) <= HISTORY_SUMMARY_MAX_CHARS

    def test_generate_draft_injects_history_prefix(self, monkeypatch):
        """有摘要时，生成器 user prompt 应带历史段；模板声明『不是证据』。"""
        from src.agent import harness

        captured: dict = {}

        def fake_chat_completion(system, user, **kwargs):
            captured["user"] = user
            return "这是一段足够长的模拟回复内容。"

        import src.llm as llm
        monkeypatch.setattr(llm, "chat_completion", fake_chat_completion)

        obs = [{"tool": "local_search",
                "result": {"snippets": [{"text": "第一条……", "title": "测试法", "score": 0.9}]}}]
        draft = harness._generate_draft("那第二款呢？", obs, [], session_summary="用户: 第一条是什么")
        assert draft.startswith("这是一段足够长")
        assert "会话历史摘要" in captured["user"]
        assert "不是证据" in captured["user"]
        assert captured["user"].index("会话历史摘要") < captured["user"].index("用户问题")

    def test_generate_draft_without_summary_has_no_history(self, monkeypatch):
        from src.agent import harness

        captured: dict = {}

        def fake_chat_completion(system, user, **kwargs):
            captured["user"] = user
            return "这是一段足够长的模拟回复内容。"

        import src.llm as llm
        monkeypatch.setattr(llm, "chat_completion", fake_chat_completion)

        harness._generate_draft("问题", [], [])
        assert "会话历史摘要" not in captured["user"]
