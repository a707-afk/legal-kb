"""Harness V2 Phase 3：会话上下文分层优化的回归测试。

覆盖六块：
1. 分层加载：RECENT_TURNS 轮以内只有 Layer 1 原文，无 Layer 2 摘要
2. 惰性触发：超阈值 + 含指代词 + 预算够 → 触发摘要
3. 惰性跳过：超阈值但输入自包含（无指代词）→ 不触发摘要
4. 增量摘要：第二次触发复用旧摘要 + 新内容
5. 并发锁：同 session_id 的并发请求被串行化
6. context_layers_used：审计元数据正确记录使用了哪些层
"""
from __future__ import annotations

import asyncio

import pytest

from src.agent.state_machine import HarnessState


@pytest.fixture(autouse=True)
def _clean_sessions():
    """每个用例前后清空会话注册表 + 锁表，避免跨用例串话。"""
    from src.session_mgr import clear_sessions
    clear_sessions()
    yield
    clear_sessions()


def _make_ctx(session_id, objective="那个之前说的条款还适用吗？"):
    """构造一个最小可用的 HarnessContext（仅填充 _handle_context_load 用到的字段）。"""
    from src.agent.harness import HarnessContext
    return HarnessContext(
        run_id="run-test",
        objective=objective,
        tenant_id="t1",
        user_id="u1",
        user_context={},
        session_id=session_id,
        ticket_id=None,
        t_start=0.0,
        budget={},
        run=None,
    )


# ── 1. 分层加载 ────────────────────────────────────────────────────────────

class TestLayeredLoading:
    async def test_within_recent_turns_only_raw_no_summary(self):
        """3 轮以内：Layer 1 原文加载，Layer 2 摘要不触发。"""
        from src.agent import harness
        from src.session_mgr import record_turn

        record_turn("s1", user_message="劳动合同法第几条？", assistant_message="第四十六条……")
        record_turn("s1", user_message="那第四十七条呢？", assistant_message="第四十七条……")

        ctx = _make_ctx("s1", objective="那个第四十七条怎么理解？")
        state = await harness._handle_context_load(ctx)

        assert state == HarnessState.RISK_INTENT
        assert ctx.session_raw  # Layer 1 有原文
        assert "劳动合同法" in ctx.session_raw
        assert ctx.session_summary == ""  # Layer 2 未触发（历史未超 3 轮）
        assert ctx.context_layers_used == ["raw_recent"]

    async def test_no_session_id_empty_layers(self):
        from src.agent import harness
        ctx = _make_ctx(None)
        state = await harness._handle_context_load(ctx)
        assert state == HarnessState.RISK_INTENT
        assert ctx.session_raw == ""
        assert ctx.session_summary == ""
        assert ctx.context_layers_used == []

    async def test_unknown_session_empty_layers(self):
        from src.agent import harness
        ctx = _make_ctx("never-seen")
        await harness._handle_context_load(ctx)
        assert ctx.session_raw == ""
        assert ctx.session_summary == ""
        assert ctx.context_layers_used == []

    def test_get_recent_raw_formats_and_truncates(self):
        from src.session_mgr import get_or_create_session
        mem = get_or_create_session("s")
        mem.add_message("user", "问" * 300)
        mem.add_message("assistant", "答" * 300)
        raw = mem.get_recent_raw()
        assert raw.startswith("用户: ")
        assert "助手: " in raw
        # 单条截断 200 字
        for line in raw.splitlines():
            assert len(line) <= len("用户: ") + 200


# ── 2 & 3 & 6. 惰性触发 / 跳过 / 审计元数据 ──────────────────────────────

class TestLazySummaryTrigger:
    async def test_triggers_when_over_threshold_with_reference(self, monkeypatch):
        """超 3 轮 + 含指代词 + 预算够 → 触发摘要，两层都用上。"""
        import src.llm as llm
        from src.agent import harness
        from src.session_mgr import record_turn

        calls: list[str] = []

        def fake_chat(system, user, **kwargs):
            calls.append(user)
            return "当事人咨询劳动合同解除的经济补偿问题。"

        monkeypatch.setattr(llm, "chat_completion", fake_chat)

        for i in range(4):  # 8 条消息 > RECENT_TURNS*2=6
            record_turn("s2", user_message=f"问题{i}", assistant_message=f"回答{i}")

        ctx = _make_ctx("s2", objective="那个之前说的怎么算？")
        await harness._handle_context_load(ctx)

        assert calls, "摘要 LLM 应被调用"
        assert ctx.session_summary == "当事人咨询劳动合同解除的经济补偿问题。"
        assert ctx.context_layers_used == ["raw_recent", "lazy_summary"]

    async def test_skips_when_input_self_contained(self, monkeypatch):
        """超 3 轮但输入自包含（无指代词）→ 不触发摘要，不调 LLM。"""
        import src.llm as llm
        from src.agent import harness
        from src.session_mgr import record_turn

        calls: list[str] = []
        monkeypatch.setattr(
            llm, "chat_completion",
            lambda system, user, **kw: calls.append(user) or "不该被用到",
        )

        for i in range(4):
            record_turn("s3", user_message=f"问题{i}", assistant_message=f"回答{i}")

        ctx = _make_ctx("s3", objective="合同法第52条的内容是什么")
        await harness._handle_context_load(ctx)

        assert calls == [], "自包含问题不应触发摘要 LLM"
        assert ctx.session_summary == ""
        assert ctx.context_layers_used == ["raw_recent"]

    async def test_skips_when_token_budget_tight(self):
        """预算不足（<1500）→ 即使有指代词也不触发。"""
        from src.session_mgr import get_or_create_session
        mem = get_or_create_session("s4")
        for i in range(4):
            mem.add_message("user", f"问题{i}")
            mem.add_message("assistant", f"回答{i}")
        # 预算 1000 < 1500
        assert mem.get_or_generate_summary("那个之前说的", 1000) == ""

    def test_estimate_token_remaining_positive_for_short_input(self):
        from src.agent.harness import _estimate_token_remaining
        ctx = _make_ctx("s", objective="简短问题")
        assert _estimate_token_remaining(ctx) > 1500


# ── 4. 增量摘要 ────────────────────────────────────────────────────────────

class TestIncrementalSummary:
    def test_second_trigger_reuses_old_summary(self, monkeypatch):
        import src.llm as llm
        from src.session_mgr import get_or_create_session

        captured: list[str] = []

        def fake_chat(system, user, **kwargs):
            captured.append(user)
            return f"摘要V{len(captured)}"

        monkeypatch.setattr(llm, "chat_completion", fake_chat)

        mem = get_or_create_session("s5")
        for i in range(4):  # 8 条消息
            mem.add_message("user", f"问题{i}")
            mem.add_message("assistant", f"回答{i}")

        s1 = mem.get_or_generate_summary("那个之前说的", 10000)
        assert s1 == "摘要V1"
        assert "旧摘要" not in captured[0]  # 第一次无旧摘要可复用

        # 再追加一轮 → 触发增量更新
        mem.add_message("user", "问题4")
        mem.add_message("assistant", "回答4")
        s2 = mem.get_or_generate_summary("那个继续说", 10000)
        assert s2 == "摘要V2"
        assert "旧摘要: 摘要V1" in captured[1]  # 第二次复用旧摘要 + 新内容

    def test_summary_capped_at_max_chars(self, monkeypatch):
        import src.llm as llm
        from src.session_mgr import SessionMemory, get_or_create_session

        monkeypatch.setattr(llm, "chat_completion", lambda s, u, **k: "长" * 2000)
        mem = get_or_create_session("s6")
        for i in range(4):
            mem.add_message("user", f"问题{i}")
            mem.add_message("assistant", f"回答{i}")
        summary = mem.get_or_generate_summary("那个之前说的", 10000)
        assert len(summary) <= SessionMemory.SUMMARY_MAX_CHARS

    def test_summary_llm_failure_returns_empty(self, monkeypatch):
        """摘要 LLM 抛异常时返回空串，不阻断主流程。"""
        import src.llm as llm
        from src.session_mgr import get_or_create_session

        def boom(*a, **k):
            raise RuntimeError("LLM down")

        monkeypatch.setattr(llm, "chat_completion", boom)
        mem = get_or_create_session("s7")
        for i in range(4):
            mem.add_message("user", f"问题{i}")
            mem.add_message("assistant", f"回答{i}")
        assert mem.get_or_generate_summary("那个之前说的", 10000) == ""


# ── 5. 并发锁 ──────────────────────────────────────────────────────────────

class TestSessionLock:
    def test_same_session_returns_same_lock(self):
        from src.session_mgr import get_session_lock
        assert get_session_lock("abc") is get_session_lock("abc")

    def test_different_session_returns_different_lock(self):
        from src.session_mgr import get_session_lock
        assert get_session_lock("a") is not get_session_lock("b")

    def test_lock_lru_eviction(self):
        import src.session_mgr as sm
        for i in range(sm._MAX_LOCKS + 10):
            sm.get_session_lock(f"sid{i}")
        assert len(sm._session_locks) == sm._MAX_LOCKS

    async def test_concurrent_requests_serialized(self):
        """同 session_id 的两个并发请求被串行化（无交错）。"""
        from src.session_mgr import get_session_lock

        order: list[str] = []

        async def worker(name: str):
            lock = get_session_lock("shared")
            async with lock:
                order.append(f"{name}-start")
                await asyncio.sleep(0.02)
                order.append(f"{name}-end")

        await asyncio.gather(worker("A"), worker("B"))

        # 串行化：一个 worker 完整跑完（start→end）后另一个才开始
        assert order[0].endswith("-start")
        assert order[1].endswith("-end")
        assert order[0][0] == order[1][0]
        assert order[2].endswith("-start")
        assert order[3].endswith("-end")


# ── 草稿注入：Layer 1 + Layer 2 都进 prompt ────────────────────────────────

class TestDraftInjection:
    def test_history_block_contains_both_layers(self):
        from src.agent.harness import _build_history_block
        block = _build_history_block("旧摘要内容", "用户: 问\n助手: 答")
        assert "会话历史摘要" in block  # HISTORY_SUMMARY_SECTION 抬头
        assert "旧摘要内容" in block
        assert "最近对话原文" in block
        assert "用户: 问" in block

    def test_history_block_empty_when_no_history(self):
        from src.agent.harness import _build_history_block
        assert _build_history_block("", "") == ""

    def test_history_block_bounded(self):
        from src.agent.harness import HISTORY_SUMMARY_MAX_CHARS, _build_history_block
        block = _build_history_block("摘" * 2000, "原文" * 2000)
        # 截断到上限 + 结尾两个换行
        assert len(block) <= HISTORY_SUMMARY_MAX_CHARS + 2

    def test_generate_draft_injects_raw_and_summary(self, monkeypatch):
        from src.agent import harness

        captured: dict = {}

        def fake_chat(system, user, **kwargs):
            captured["user"] = user
            return "这是一段足够长的模拟回复内容。"

        import src.llm as llm
        monkeypatch.setattr(llm, "chat_completion", fake_chat)

        obs = [{"tool": "local_search",
                "result": {"snippets": [{"text": "第一条……", "title": "测试法", "score": 0.9}]}}]
        harness._generate_draft(
            "那第二款呢？", obs, [],
            session_summary="用户问过第一条", session_raw="用户: 第一条是什么\n助手: ……",
        )
        assert "会话历史摘要" in captured["user"]
        assert "最近对话原文" in captured["user"]
        # 历史块在用户问题之前
        assert captured["user"].index("会话历史摘要") < captured["user"].index("用户问题")


# ── 向后兼容 ───────────────────────────────────────────────────────────────

class TestBackwardCompat:
    def test_summarize_last_n_still_works(self):
        from src.session_mgr import get_or_create_session
        mem = get_or_create_session("s8")
        mem.add_message("user", "劳动合同法第几条？")
        mem.add_message("assistant", "第四十六条")
        s = mem.summarize_last_n(3)
        assert "劳动合同法" in s
        assert "第四十六条" in s
