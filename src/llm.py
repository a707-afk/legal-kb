"""LLM 客户端：**单一后端 SenseNova（商汤）**。

选型定案（2026-09-28，docs/00 D-24）
----------------------------------
统一使用 `sensenova-6.8-flash-lite`，**废弃智谱 glm 路线**。

为什么统一（而不是保留双后端可切换）：
1. **避免"模型不一样导致错误"**——双后端意味着同一份评测可能跑在不同模型上，
   指标不可比。这正是旧项目"E2 与基线跑在不同版本上"那类事故的翻版。
2. **延迟**：可复跑对照（`scripts/bench_llm_backends.py`，6 题）实测 p50
   `glm-4.5-flash` **32.5s** / `glm-4-flash` **19.4s** / 商汤 `sensenova-6.8-flash-lite` **4.8s**。
   单次实测的"glm-4-flash 1.0s"是**单样本假象**，样本量上去后不成立。
3. **引用精度**：商汤 `sensenova-6.8-flash-lite` 的条号可精确到"第（二）项"，与 glm-4.5-flash 同级。

`LLM_BACKEND` 仍保留为配置项，但**只接受 `sensenova`**；设成别的值会直接报错
（不静默回退——静默回退正是本项目反复强调要避免的失败模式）。

凭据
----
`SENSENOVA_API_KEYS`（**复数**，逗号分隔多个 key，自动轮转）。
只从环境变量 / `.env`（已 gitignore）读，`.env.example` 只放占位符。
遇到 429 会**自动换下一个 key** 并拉长全局限流间隔。

`reasoning_content`
-------------------
商汤把思维链放在独立的 `reasoning_content` 字段（`completion_tokens_details.reasoning_tokens` 可见），
`chat_completion_full()` 会单独带出来，便于 D-15 的 trace 记录。
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from collections import deque
from typing import Any

from src.config_LEGACY_REFERENCE import get_settings

logger = logging.getLogger(__name__)

# ── 后端常量（单一后端，不再分派）──────────────────────────────────
BACKEND_NAME = "sensenova"
SENSENOVA_BASE = "https://token.sensenova.cn/v1"
DEFAULT_MODEL = "sensenova-6.8-flash-lite"
API_KEY_ENV = "SENSENOVA_API_KEYS"

# 向后兼容：旧代码/旧测试引用的模块级符号
_SENSENOVA_KEYS: list[str] = []
_key_idx = 0


def get_backend_name() -> str:
    """校验并返回后端名。只允许 `sensenova`；其他值直接报错，不静默回退。"""
    try:
        configured = (getattr(get_settings(), "llm_backend", "") or BACKEND_NAME).strip().lower()
    except Exception:
        configured = BACKEND_NAME
    if configured != BACKEND_NAME:
        raise RuntimeError(
            f"LLM_BACKEND={configured!r} 已废弃。本项目于 2026-09-28 统一到 "
            f"{BACKEND_NAME!r}（模型 {DEFAULT_MODEL}），理由见 docs/00 决策 D-24。"
            f"请把 .env 改为 LLM_BACKEND={BACKEND_NAME} 或删除该行（不静默回退）。"
        )
    return BACKEND_NAME


def _load_keys() -> list[str]:
    """读取 SenseNova API keys（环境变量优先，`.env` 兜底），逗号分隔。"""
    raw = (os.getenv(API_KEY_ENV) or "").strip()
    if not raw:
        try:
            raw = str(getattr(get_settings(), "sensenova_api_keys", "") or "").strip()
        except Exception:  # pragma: no cover
            raw = ""
    return [k.strip() for k in raw.split(",") if k.strip()]


class LLMNotConfigured(RuntimeError):
    """凭据/配置缺失 —— **永久性错误，不重试**（重试只是白等）。"""


def _next_key() -> str:
    """取一个 key；多 key 时轮转（429 重试时自然换到下一个 key）。"""
    global _key_idx, _SENSENOVA_KEYS
    keys = _load_keys()
    if not keys:
        raise LLMNotConfigured(
            f"未配置 {API_KEY_ENV}，无法调用 LLM。"
            f"请设置环境变量：export {API_KEY_ENV}='sk-aaa,sk-bbb,sk-ccc'"
            f"（多个 key 逗号分隔，会自动轮转；也可写入已 gitignore 的 .env）"
        )
    _SENSENOVA_KEYS = keys
    idx = _key_idx % len(keys)
    _key_idx = (idx + 1) % len(keys)
    return keys[idx]


def get_model_name() -> str:
    """当前使用的模型名。"""
    try:
        configured = (getattr(get_settings(), "llm_model", "") or "").strip()
    except Exception:
        configured = ""
    return configured or DEFAULT_MODEL


def get_base_url() -> str:
    return SENSENOVA_BASE


def _get_client():
    """获取同步 OpenAI 兼容客户端，返回 ``(client, model)``。"""
    from openai import OpenAI

    return (
        OpenAI(api_key=_next_key(), base_url=SENSENOVA_BASE),
        get_model_name(),
    )


def _get_async_client():
    """获取异步 OpenAI 兼容客户端，返回 ``(client, model)``。"""
    from openai import AsyncOpenAI

    return (
        AsyncOpenAI(api_key=_next_key(), base_url=SENSENOVA_BASE),
        get_model_name(),
    )


# ── 限流与退避 ─────────────────────────────────────────────────────

def _coerce_number(value: Any, default: float, lo: float, hi: float) -> float:
    """把配置值收敛成 ``[lo, hi]`` 内的数值；类型不对就用默认值。

    ⚠️ 为什么不能直接 ``float(value)``：`MagicMock` 实现了 ``__float__``/``__int__``
    并返回 **1.0/1**——测试或异常配置下会把限流阈值悄悄变成 1
    （实测：RPM 上限被当成 1，3 次重试等了 180 秒）。
    这里要求值必须是真正的 ``int``/``float``（``bool`` 除外）。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(v, hi))


class _RateLimiter:
    """全局滑窗限流：① 最小间隔 ② 每分钟请求数（RPM）上限。

    为什么需要两层：实测 2026-09-28 连续调用约 12 次即报
    `429 rpm exhausted`——**光靠最小间隔不够**，商汤免费档的 rpm 上限很低。
    RPM 用 60 秒滑窗计数；触顶时等到窗口内最早的调用滑出再放行。
    遇到 429 还会**自适应拉长最小间隔**，让后续调用自然变慢。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._extra_delay = 0.0
        self._recent: deque[float] = deque()  # 最近 60s 内的调用时刻（单调钟）

    def _base_interval(self) -> float:
        try:
            raw = getattr(get_settings(), "llm_min_interval_seconds", 1.0)
        except Exception:
            raw = 1.0
        return _coerce_number(raw, default=1.0, lo=0.0, hi=60.0)

    def _max_rpm(self) -> int:
        try:
            raw = getattr(get_settings(), "llm_max_rpm", 10)
        except Exception:
            raw = 10
        return int(_coerce_number(raw, default=10.0, lo=1.0, hi=600.0))

    def wait(self) -> None:
        """结算一次调用配额；需要等待时在**锁外** sleep（不阻塞其它线程的计数）。

        ⚠️ 刻意**不用 while 循环**重试：若 ``time.sleep`` 被替换成 no-op
        （测试或异常环境），循环会因为时间不前进而自旋死锁。
        单次结算已足够——等待时长按"窗口内最早调用滑出"精确计算。
        """
        with self._lock:
            now = time.monotonic()
            # 滑出 60s 窗口
            while self._recent and now - self._recent[0] >= 60.0:
                self._recent.popleft()

            rpm_wait = 0.0
            limit = self._max_rpm()
            if limit > 0 and len(self._recent) >= limit:
                rpm_wait = 60.0 - (now - self._recent[0]) + 0.05

            interval = self._base_interval() + self._extra_delay
            gap = now - self._last_call
            interval_wait = interval - gap if gap < interval else 0.0

            sleep_for = max(rpm_wait, interval_wait)
            # 把本次调用"预约"在 sleep 之后，避免并发线程同时醒来挤在一起
            target = now + sleep_for
            self._recent.append(target)
            self._last_call = target

        if sleep_for > 0:
            time.sleep(sleep_for)

    def note_rate_limited(self) -> None:
        """被限流后把最小间隔翻倍（上限 10s）。"""
        with self._lock:
            self._extra_delay = min(max(self._extra_delay * 2, 0.5), 10.0)

    def note_success(self) -> None:
        """连续成功则缓慢回收额外间隔。"""
        with self._lock:
            if self._extra_delay > 0:
                self._extra_delay = max(self._extra_delay * 0.5, 0.0)
                if self._extra_delay < 0.1:
                    self._extra_delay = 0.0

    @property
    def current_interval(self) -> float:
        with self._lock:
            return self._base_interval() + self._extra_delay

    @property
    def calls_in_window(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._recent and now - self._recent[0] >= 60.0:
                self._recent.popleft()
            return len(self._recent)


_RATE_LIMITER = _RateLimiter()


def _is_rate_limited(err_str: str) -> bool:
    low = (err_str or "").lower()
    return any(tok in low for tok in (
        "429", "rate limit", "ratelimit", "rpm", "tpm", "quota",
        "too many requests", "exhausted", "限流", "超出",
    ))


def _retry_wait(attempt: int) -> float:
    """指数退避 + 抖动。抖动避免并发请求同时重试、再次打爆配额。"""
    try:
        settings = get_settings()
        base = _coerce_number(getattr(settings, "llm_backoff_base_seconds", 2.0), 2.0, 0.1, 120.0)
        cap = _coerce_number(getattr(settings, "llm_backoff_max_seconds", 30.0), 30.0, 1.0, 300.0)
        jitter = _coerce_number(getattr(settings, "llm_backoff_jitter_seconds", 1.0), 1.0, 0.0, 30.0)
    except Exception:
        base, cap, jitter = 2.0, 30.0, 1.0
    wait = min(base * (2 ** attempt), cap)
    return wait + (random.uniform(0, jitter) if jitter > 0 else 0.0)


def _is_retryable(err_str: str, attempt: int, max_retries: int) -> bool:
    return (_is_rate_limited(err_str) or "503" in err_str or "502" in err_str
            or "timeout" in err_str.lower() or attempt < max_retries)


def _retryable_exc(exc: Exception, attempt: int, max_retries: int) -> bool:
    """永久性配置错误不重试；其余按 `_is_retryable` 判断。"""
    if isinstance(exc, LLMNotConfigured):
        return False
    return _is_retryable(str(exc), attempt, max_retries)


# ── 对外接口 ───────────────────────────────────────────────────────

def _extract_reasoning(choice: Any) -> str:
    """取出思维链文本。

    商汤把思维链放在**非标准字段** `reasoning` 里（OpenAI SDK 会把它塞进
    `model_extra`，因此 `getattr(choice, "reasoning")` 取得到，而
    `choice.reasoning_content` 是空）。实测 2026-09-29：`usage.reasoning_tokens=77`
    但 `reasoning_content` 为 None —— 旧代码读错字段，思维链被静默丢弃。
    这里按 `reasoning` → `reasoning_content` → `model_extra` 顺序兜底，
    避免以后后端改字段名又静默丢数据。
    """
    for attr in ("reasoning", "reasoning_content"):
        v = getattr(choice, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    extra = getattr(choice, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        v = extra.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def chat_completion_full(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """调用 LLM 并返回结构化结果（含 reasoning / usage / 延迟），供 trace 使用。

    Returns:
        ``{"content", "reasoning", "model", "backend", "usage", "latency_ms", "attempts"}``
    """
    settings = get_settings()
    max_retries = getattr(settings, "llm_max_retries", 2)
    timeout_s = getattr(settings, "llm_timeout_seconds", 60.0)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        _RATE_LIMITER.wait()
        t0 = time.perf_counter()
        try:
            client, model = _get_client()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "timeout": timeout_s,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0].message
            latency_ms = (time.perf_counter() - t0) * 1000
            _RATE_LIMITER.note_success()
            usage = getattr(resp, "usage", None)
            return {
                "content": (choice.content or "").strip(),
                "reasoning": _extract_reasoning(choice),
                "model": model,
                "backend": BACKEND_NAME,
                "usage": usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None,
                "latency_ms": round(latency_ms, 1),
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if _is_rate_limited(err_str):
                _RATE_LIMITER.note_rate_limited()
            if _retryable_exc(exc, attempt, max_retries):
                wait = _retry_wait(attempt)
                logger.warning(
                    "LLM attempt %d/%d failed: %s, retrying in %.1fs (下一个 key)",
                    attempt + 1, max_retries + 1, str(exc)[:200], wait,
                )
                time.sleep(wait)
            else:
                break

    if isinstance(last_exc, LLMNotConfigured):
        raise last_exc  # 保留类型，便于调用方区分"没配 key"与"调用失败"
    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_exc}") from last_exc


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    """调用 LLM 聊天补全，返回纯文本（`reasoning_content` 已剥离）。"""
    return chat_completion_full(system_prompt, user_prompt)["content"]


async def achat_completion(system_prompt: str, user_prompt: str) -> str:
    """异步 LLM 聊天补全 —— 非阻塞，使用 AsyncOpenAI + asyncio.sleep。"""
    settings = get_settings()
    max_retries = getattr(settings, "llm_max_retries", 2)
    timeout_s = getattr(settings, "llm_timeout_seconds", 60.0)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        await asyncio.to_thread(_RATE_LIMITER.wait)  # 限流不阻塞事件循环
        try:
            client, model = _get_async_client()
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=timeout_s,
            )
            choice = resp.choices[0].message
            _RATE_LIMITER.note_success()
            return (choice.content or "").strip()
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if _is_rate_limited(err_str):
                _RATE_LIMITER.note_rate_limited()
            if _retryable_exc(exc, attempt, max_retries):
                wait = _retry_wait(attempt)
                logger.warning(
                    "LLM async attempt %d/%d failed: %s, retrying in %.1fs",
                    attempt + 1, max_retries + 1, str(exc)[:200], wait,
                )
                await asyncio.sleep(wait)
            else:
                break

    if isinstance(last_exc, LLMNotConfigured):
        raise last_exc
    raise RuntimeError(f"LLM async call failed after {max_retries + 1} attempts: {last_exc}") from last_exc


def get_key_pool_size() -> int:
    """当前配置了几个 key（用于自检与报告）。"""
    return len(_load_keys())


# ── 向后兼容 ───────────────────────────────────────────────────────
def get_zhipu_client():
    """向后兼容旧调用点：返回当前（SenseNova）客户端。"""
    client, _ = _get_client()
    return client
