"""Global test configuration – disables API auth and resets settings cache."""
from __future__ import annotations

import os

# Set environment variables BEFORE any app imports
os.environ.setdefault("API_AUTH_ENABLED", "false")
os.environ.setdefault("API_KEYS", "test-key-for-ci")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear settings lru_cache before and after each test so env overrides apply."""
    from src.config_LEGACY_REFERENCE import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_trace_dir(tmp_path_factory, monkeypatch):
    """把 TRACE_DIR 指向临时目录，禁止任何测试写入真实的 data/traces/。

    Harness Phase 4 起，状态机审计事件与 metrics 会经 src.trace.traces_root()
    落盘（默认 data/traces/）。没有这道全局隔离，跑一次全量测试就会在仓库里
    留下几十个 {run_id}.audit.jsonl。需要精确控制落盘位置的测试（如
    tests/test_trace.py、tests/agent/test_state_machine.py）可以再用
    monkeypatch.setenv("TRACE_DIR", ...) 覆盖本 fixture。
    """
    monkeypatch.setenv("TRACE_DIR", str(tmp_path_factory.mktemp("traces")))
    yield
