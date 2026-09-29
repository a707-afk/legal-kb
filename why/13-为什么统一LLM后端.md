# 13 统一 LLM 后端（查询·生成这一环）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：`/chat` 的生成步 + Agent 的规划/生成/评估，都调 `src/llm.py`。
> 这篇讲**LLM 客户端怎么造的、为什么锁死单一后端**。

## 1. 它在系统里是什么

- `src/llm.py` 是全项目**唯一**的 LLM 出口：`chat_completion(system, user)` / `achat_completion(...)`。
- 后端锁定 **SenseNova（商汤）** 的 OpenAI 兼容端点；`LLM_BACKEND` 只接受 `sensenova`，设别的**直接报错**（不静默回退）。
- 谁调它：`routes_rag._execute_chat`（生成答案）、`harness` 的 `_build_plan`/`_generate_draft`/`_evaluate_result`、`agent_grader`、`research_tools.synthesize`。

## 2. 配置/接口长什么样

```
SENSENOVA_API_KEYS=sk-aaa,sk-bbb     # 逗号分隔多 key，自动轮转
LLM_BACKEND=sensenova                # 只认这个值
```
`.env` 里 key 只放占位符，真 key 走环境变量（`.env` 已 gitignore）。

## 3. 代码怎么做的（`src/llm.py`）

- `_load_keys()` / `_next_key()`：从 settings/env 读多 key，**轮转**（`_key_idx` 循环）。
- `chat_completion()`：`for attempt in range(max_retries+1)` 重试；命中 `429/502/503` 或未到上限就**指数退避**
  （`wait = 2**attempt`）后重试；全失败才 raise。异步版 `achat_completion` 用 `AsyncOpenAI` + `asyncio.sleep`。
- 选型脚本 `scripts/bench_llm_backends.py`：跑 **6 题对照**（延迟 p50 + 条号精度 + 给出项号的题数）。

## 4. 为什么必须这么造（工程约束）

- **评测可比性是底线**：双后端意味着同一份评测可能跑在不同模型上 → 差值里混着模型差异 → 结论作废。
  旧项目就栽在"E2 与基线跑在不同版本上"。所以锁死单一后端，`LLM_BACKEND` 设错直接报错。
- **多 key 轮转**是为了扛限流（429 自动换下一个 + 拉长退避），不是为了"多后端容灾"。
- **一个方法论坑（比结论更重要）**：旧文档记 "glm-4-flash 1.0s" 是**单次测量**；样本提到 6 题后 p50 是 **19.4s**——
  首次请求的缓存/冷启动会让单样本延迟严重失真。所以选型必须多样本（呼应 04：别用会被污染的数字下结论）。

## 5. 怎么验证它对

- `python scripts/bench_llm_backends.py`：看各候选模型的延迟 p50 / 条号精度 / 给项号数（多样本）。
- `pytest tests/test_llm_client.py tests/test_llm_async.py`：无 key 时报错、异步重试逻辑（client 被 mock，不打真 API）。
- ⚠️ 待核对：`docs/06 §四` 定案写 `sensenova-6.8-flash-lite`，而 `src/llm.py::_SENSENOVA_MODEL` 是 `deepseek-v4-flash`
  （同一 SenseNova 端点下的不同型号）——两处口径要对齐。

## 6. 代码位置

- `src/llm.py`（`_next_key` 轮转 / `chat_completion` / `achat_completion` 退避重试）、`scripts/bench_llm_backends.py`
- 正式账本：`docs/00-决策记录.md` D-24；实测 `docs/06-环境与模型选型.md` §四
