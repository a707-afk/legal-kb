"""LLM 客户端测试：单一后端（SenseNova）+ 凭据/限流契约。

背景（docs/00 D-24）：2026-09-28 定案统一到 `sensenova-6.8-flash-lite`，
**废弃智谱 glm 路线**。原因是双后端会让同一份评测跑在不同模型上 → 指标不可比。
本文件锁住三条契约：
1. 只接受 `LLM_BACKEND=sensenova`，其他值**报错而非静默回退**
2. 无 API key 必须报 `RuntimeError`，且消息里点名 `SENSENOVA_API_KEYS`
3. 多 key 逗号分隔可轮转；`reasoning_content` 与 `content` 分离
"""
from __future__ import annotations

import sys
import time
from collections import deque
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestSingleBackend(unittest.TestCase):
    def test_backend_name_is_sensenova(self):
        import src.llm as llm_mod
        self.assertEqual(llm_mod.BACKEND_NAME, "sensenova")
        self.assertEqual(llm_mod.API_KEY_ENV, "SENSENOVA_API_KEYS")
        self.assertIn("sensenova.cn", llm_mod.SENSENOVA_BASE)
        self.assertEqual(llm_mod.DEFAULT_MODEL, "sensenova-6.8-flash-lite")

    def test_deprecated_backend_raises_not_silent(self):
        """`LLM_BACKEND=zhipu` 必须报错（已废弃），不能静默回退到 sensenova。"""
        import src.llm as llm_mod
        with patch.object(llm_mod, "get_settings") as ms:
            ms.return_value = MagicMock(llm_backend="zhipu")
            with self.assertRaises(RuntimeError) as ctx:
                llm_mod.get_backend_name()
            self.assertIn("废弃", str(ctx.exception))
            self.assertIn("D-24", str(ctx.exception))

    def test_sensenova_backend_accepted(self):
        import src.llm as llm_mod
        with patch.object(llm_mod, "get_settings") as ms:
            ms.return_value = MagicMock(llm_backend="sensenova")
            self.assertEqual(llm_mod.get_backend_name(), "sensenova")


class TestKeyPool(unittest.TestCase):
    def test_multi_keys_rotate(self):
        """逗号分隔多 key 必须轮转使用。"""
        import src.llm as llm_mod
        with patch.object(llm_mod, "_load_keys", return_value=["k1", "k2", "k3"]):
            llm_mod._key_idx = 0
            seen = [llm_mod._next_key() for _ in range(4)]
        self.assertEqual(seen, ["k1", "k2", "k3", "k1"])

    def test_no_key_raises_with_env_name(self):
        import src.llm as llm_mod
        with patch.object(llm_mod, "_load_keys", return_value=[]):
            with self.assertRaises(RuntimeError) as ctx:
                llm_mod.chat_completion("sys", "user")
            self.assertIn("SENSENOVA_API_KEYS", str(ctx.exception))
            self.assertIsInstance(ctx.exception, llm_mod.LLMNotConfigured)

    def test_key_pool_size(self):
        import src.llm as llm_mod
        with patch.object(llm_mod, "_load_keys", return_value=["a", "b"]):
            self.assertEqual(llm_mod.get_key_pool_size(), 2)


class TestChatCompletion(unittest.TestCase):
    def _mock_client(self, content="答案", reasoning="思考"):
        msg = MagicMock()
        msg.content = content
        msg.reasoning_content = reasoning
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage.model_dump.return_value = {"total_tokens": 7}
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client

    def test_reasoning_separated(self):
        import src.llm as llm_mod
        client = self._mock_client()
        with patch.object(llm_mod, "_get_client", return_value=(client, llm_mod.DEFAULT_MODEL)), \
             patch.object(llm_mod, "_load_keys", return_value=["k"]):
            out = llm_mod.chat_completion_full("sys", "user")
        self.assertEqual(out["content"], "答案")
        self.assertEqual(out["reasoning"], "思考")
        self.assertEqual(out["backend"], "sensenova")
        self.assertEqual(out["usage"], {"total_tokens": 7})
        self.assertIn("latency_ms", out)

    def test_chat_completion_returns_content_only(self):
        import src.llm as llm_mod
        client = self._mock_client(content="只有答案", reasoning="不该出现")
        with patch.object(llm_mod, "_get_client", return_value=(client, "m")), \
             patch.object(llm_mod, "_load_keys", return_value=["k"]):
            out = llm_mod.chat_completion("sys", "user")
        self.assertEqual(out, "只有答案")


class TestRateLimiter(unittest.TestCase):
    def test_rate_limited_grows_interval(self):
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(rl, "_base_interval", return_value=0.0):
            before = rl.current_interval
            rl.note_rate_limited()
            self.assertGreater(rl.current_interval, before)
            rl.note_rate_limited()
            self.assertGreater(rl.current_interval, 0.0)

    def test_success_recovers_interval(self):
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(rl, "_base_interval", return_value=0.0):
            for _ in range(4):
                rl.note_rate_limited()
            peak = rl.current_interval
            for _ in range(10):
                rl.note_success()
            self.assertLess(rl.current_interval, peak)

    def test_extra_delay_capped(self):
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(rl, "_base_interval", return_value=0.0):
            for _ in range(20):
                rl.note_rate_limited()
            self.assertLessEqual(rl.current_interval, 10.0)

    def test_rate_limit_detection(self):
        import src.llm as llm_mod
        self.assertTrue(llm_mod._is_rate_limited("Error code: 429"))
        self.assertTrue(llm_mod._is_rate_limited("rpm exhausted"))
        self.assertTrue(llm_mod._is_rate_limited("Rate limit reached"))
        self.assertFalse(llm_mod._is_rate_limited("Error code: 400 bad request"))

    def test_rpm_window_counts_and_slides(self):
        """RPM 滑窗：60s 内的调用计数，超窗自动滑出。"""
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(rl, "_max_rpm", return_value=3), \
             patch.object(rl, "_base_interval", return_value=0.0):
            for _ in range(3):
                rl.wait()  # 3 次都在配额内，不应阻塞
            self.assertEqual(rl.calls_in_window, 3)
            # 把窗口内的记录改成 61 秒前 → 应全部滑出
            rl._recent = deque(t - 61.0 for t in rl._recent)
            self.assertEqual(rl.calls_in_window, 0)

    def test_rpm_limit_blocks_when_exhausted(self):
        """配额用尽时必须等待，而不是硬发（实测 rpm exhausted 会浪费整轮重试）。"""
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(rl, "_max_rpm", return_value=2), \
             patch.object(rl, "_base_interval", return_value=0.0), \
             patch.object(llm_mod.time, "sleep") as mock_sleep:
            rl._recent.extend([time.monotonic() - 1.0, time.monotonic() - 1.0])
            rl.wait()
            self.assertTrue(mock_sleep.called)
            waited = mock_sleep.call_args[0][0]
            self.assertGreater(waited, 50.0)  # 需要等窗口内最早的调用滑出

    def test_no_wait_when_under_limit(self):
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(rl, "_max_rpm", return_value=100), \
             patch.object(rl, "_base_interval", return_value=0.0), \
             patch.object(llm_mod.time, "sleep") as mock_sleep:
            rl.wait()
            self.assertFalse(mock_sleep.called)

    def test_coerce_number_rejects_mock_like_values(self):
        """非数值配置必须回落到默认值。

        实测事故：测试里用 `MagicMock` 当 settings，`int(MagicMock())` 返回 **1**，
        于是 RPM 上限被当成 1（每分钟 1 次）→ 3 次重试等了 180 秒。
        """
        import src.llm as llm_mod
        m = MagicMock()
        self.assertEqual(llm_mod._coerce_number(m, default=10.0, lo=1.0, hi=600.0), 10.0)
        self.assertEqual(llm_mod._coerce_number(None, default=10.0, lo=1.0, hi=600.0), 10.0)
        self.assertEqual(llm_mod._coerce_number("abc", default=10.0, lo=1.0, hi=600.0), 10.0)
        self.assertEqual(llm_mod._coerce_number(True, default=10.0, lo=1.0, hi=600.0), 10.0)
        self.assertEqual(llm_mod._coerce_number(float("nan"), default=10.0, lo=1.0, hi=600.0), 10.0)
        # 正常值 + 越界收敛
        self.assertEqual(llm_mod._coerce_number(5, default=10.0, lo=1.0, hi=600.0), 5.0)
        self.assertEqual(llm_mod._coerce_number(0.5, default=10.0, lo=1.0, hi=600.0), 1.0)
        self.assertEqual(llm_mod._coerce_number(9999, default=10.0, lo=1.0, hi=600.0), 600.0)

    def test_mock_settings_does_not_collapse_rpm_to_one(self):
        """用 MagicMock 当 settings 时，RPM 上限不能塌成 1。"""
        import src.llm as llm_mod
        rl = llm_mod._RateLimiter()
        with patch.object(llm_mod, "get_settings", return_value=MagicMock()):
            self.assertEqual(rl._max_rpm(), 10)


if __name__ == "__main__":
    unittest.main()
