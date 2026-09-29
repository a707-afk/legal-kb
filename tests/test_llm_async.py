"""Tests for llm.py async client (M1 fix)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


def _settings(**over) -> SimpleNamespace:
    """真实的数值型配置（不要用 MagicMock：它实现了 __int__/__float__ 并返回 1，
    会把限流阈值悄悄变成 1 —— 实测让本测试从 9 秒涨到 180 秒）。"""
    base = dict(
        llm_max_retries=3,
        llm_timeout_seconds=60,
        llm_max_rpm=600,
        llm_min_interval_seconds=0.0,
        llm_backoff_base_seconds=0.01,
        llm_backoff_max_seconds=0.02,
        llm_backoff_jitter_seconds=0.0,
        llm_model="test-model",
        llm_backend="sensenova",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_achat_completion_uses_async_client():
    """achat_completion must use AsyncOpenAI and not block the event loop."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Hello from async"

    async def mock_create(**kw):
        return mock_resp

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("src.llm._get_async_client", return_value=(mock_client, "test-model")), \
         patch("src.llm.get_settings", return_value=_settings()):
        from src.llm import achat_completion
        result = await achat_completion("system", "user")
        assert result == "Hello from async"


@pytest.mark.asyncio
async def test_achat_completion_retries_on_failure():
    """achat_completion must retry on 429/503 errors."""
    call_count = 0

    async def mock_create(**kw):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429 Too Many Requests")
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "OK after retry"
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("src.llm._get_async_client", return_value=(mock_client, "test-model")), \
         patch("src.llm.get_settings", return_value=_settings()):
        from src.llm import achat_completion
        result = await achat_completion("system", "user")
        assert result == "OK after retry"
        assert call_count == 3


def test_achat_completion_exists():
    """Verify achat_completion function exists and is a coroutine function."""
    from src.llm import achat_completion
    assert asyncio.iscoroutinefunction(achat_completion)
