# 06 自研 Harness 的控制循环（Agent 这一环怎么造的）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：主线三的核心。05 讲"为什么是单 Agent"，
> 这篇讲**这个单 Agent 的循环具体怎么实现、为什么不用 LangChain/LangGraph 现成的**。

## 1. 它在系统里是什么

`run_agent_harness()` 是一个**显式状态机式的循环**：规划 → 逐步执行工具 → 评估 → 改写回环 → HITL → 落盘。
每一步都 `_add_step(...)` 记成一条 `AgentStep`（可回放，见 07）。

## 2. 循环的控制流（对着代码读）

```
run_agent_harness(objective, tenant_id, ...):
  run = AgentRun(risk_level=_assess_risk(objective), ...)      # 风险评估
  if _input_injection_detection(objective).blocked: 早退        # 注入检测
  intent = _classify_intent(objective)                          # 意图（当前是中性桩）
  plan = _build_plan(objective, risk)                           # LLM 规划，失败回退 _build_plan_rule_based
  for step in plan（受 max_steps/max_latency/max_transitions 约束）:
      if step.type == "retrieve": 记 delegated_to_rag；continue
      if step.type == "execute":
          tool = registry.get(name)
          perm = check_permission(tool, user_context, params)   # 权限门
          result = await wait_for(registry.execute(...), step_timeout)   # 单步超时
          失败 → _classify_tool_failure → retryable 就退避重试   # 见 D-13
  draft = _generate_draft(...) → OutputGuard.check → _evaluate_result
  while 未通过 and rewrite < max_rewrite_attempts: 改写重生成     # ⚠️ 这个回环是"假的"，见下
  needs_human = 高风险 or 有审批 or 未通过 → HITL
  落 JSONL（get_sessionmaker）
```

## 3. 代码怎么做的（`src/agent/harness.py`）

- 预算：`DEFAULT_BUDGET`（max_steps=10 / max_tool_calls=20 / max_latency_ms / step_timeout / max_transitions=30 / max_rewrite_attempts=2）。
- 三层保护（D-14）：最大迭代（`max_transitions`）✅ / 异常回退（工具级 + run 级 try/except，落 `termination_reason`）✅ /
  **循环检测 ❌ 尚未实现**（D5 补，诚实标注）。
- 工具执行统一走 `ToolRegistry.execute`（校验→权限→幂等→执行），失败三分类在 harness 侧（见 D-13）。

## 4. 为什么必须自研（工程约束）

- **要插领域验证器**：闭环里要放"时效是否匹配""条号能否命中"这种确定性判据（可自动评测，见 04）。
  LangChain/LangGraph 的通用 Agent 循环不给这些钩子，硬塞要跟抽象搏斗。
- **每步要可判分**：`_add_step` 把 plan/execute/observe/evaluate/approve 全落成结构化步骤，
  才能"每个决策点单独开关、单独统计触发次数"。
- **一代用过 LangGraph 又砍了**（`routes_agent.py` 原 docstring 写 "LangGraph path removed"）——是有决策史的。

## 5. 怎么验证它对

- `pytest tests/agent/test_agent_harness.py`：离线跑通一条 run（LLM mock 掉）、预算/权限/失败三分类/JSONL 落盘。
- ⚠️ **已知缺陷（诚实）**：改写回环复用同一批 `all_observations`、只换生成用的 query 文本、**没有重新检索**，
  所以评估结果不变、最终走到 `forced_pass`。**D5 必须新建一条真正接回 Retriever 的路径**（见 `src/agent/__init__.py` 注释）。

## 6. 代码位置

- `src/agent/harness.py`（主循环 + 各 `_*` 辅助）、`tool_registry.py`、`permission_gate.py`
- 正式账本：`BLUEPRINT.md` D-12（控制循环）/ D-13（工具调度）/ D-14（三层保护）
