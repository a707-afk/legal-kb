# 05 不做多 Agent：Agent 层实际是什么（Agent 这一层的边界）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：主线三（Agent 流程）。
> 这篇讲**系统里 Agent 层实际长什么样**——只有一个 Harness + 两个工具，没有多 Agent。

## 1. 它在系统里是什么

- Agent 层 = **单个 `run_agent_harness` 控制循环** + **2 个只读工具**（`local_search` / `synthesize`）。
- 没有 planner-agent / retriever-agent / grader-agent 这种多实体拆分，也没有 LangGraph 图。
- 入口：`POST /agent/run` → `run_agent_harness(objective, ...)`。

## 2. 循环长什么样（数据/控制流）

```
run_agent_harness:
  风险评估 → 注入检测 → 意图分类 → 规划(_build_plan)
  → for 每个 plan step:
       工具调用经 ToolRegistry.execute：[参数校验 → 权限门 → 幂等 → 执行]
       失败 → 三分类(retryable/non_retryable/needs_human) + 退避重试
  → 生成草稿 → OutputGuard → 评估 → 改写回环 → HITL 判定 → 落 JSONL
```
预算护栏：`max_steps / max_tool_calls / max_latency_ms / step_timeout / max_transitions`（都在 `DEFAULT_BUDGET`）。

## 3. 代码怎么做的

- `src/agent/harness.py`：`run_agent_harness` 主循环 + `_build_plan`（LLM 规划，失败回退规则）+
  `_assess_risk` / `_input_injection_detection` / `_classify_tool_failure` / `_evaluate_result`。
- `src/agent/tool_registry.py`：`ToolRegistry` + `ToolDef`（side_effect / risk_level / 幂等键 / 超时）。
  只注册 `local_search`（调检索主线）+ `synthesize`（调 LLM 综合），都是 READ_ONLY / LOW。
- `src/agent/permission_gate.py`：风险分级 → allow / allow_audit / need_scope / deny。
- **已删的多 Agent/一代残留**：`src/agent/tools.py`（customer/ticket 工具）、LangGraph 路径、`agent_multi_agent_enabled` 配置。

## 4. 为什么必须这么造（工程约束，不是"不会做"）

- **多 Agent 协作没有 ground truth → 无法自动判分**。本项目立身之本是"每环有确定性验证器、可程序化评测"
  （见 04）。多 Agent 的"协作得好不好"只能人看，和这个前提冲突。
- **单任务检索-生成链，拆分收益 < 协调成本**：状态传递、失败归因、循环风险都是净负担。
- **编排能力不靠多 Agent 实体体现**：单 Harness 里的多步决策 + 工具调度 + 失败三分类 + 预算/超时/HITL
  本身就是编排，且**每个决策点可单独开关、单独评测**（这恰恰是多 Agent 难做到的）。

## 5. 怎么验证它对

- `pytest tests/agent/test_agent_harness.py`：工具注册（只有 local_search/synthesize、都 READ_ONLY/LOW）、
  权限门、失败三分类 + 真重试（`TestToolRetryClassification`）、JSONL 落盘、离线跑通一条 run。
- 反例检查：全仓 `grep -ri "langgraph\|multi_agent\|create_ticket"` 应为空（一代多 Agent/客服残留已清）。

## 6. 代码位置

- `src/agent/harness.py` / `tool_registry.py` / `permission_gate.py` / `research_tools.py` / `state.py`
- `src/api/routes_agent.py`（`/agent/run`）
- 正式账本：`BLUEPRINT.md` 第五章「明确不做」；数据论证 `docs/08-工程能力取舍.md`
