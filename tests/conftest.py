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
