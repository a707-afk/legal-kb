# 14 删掉 OPA / Redis / Prometheus / 多租户（全局·边界）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：不属某一环，是**贯穿全局的"不做什么"**。
> 这篇讲**哪些企业级组件被删了、换成了什么真在用的轻量实现、删的工程理由**。

## 1. 它在系统里是什么

代码从旧企业项目 fork 来，带了一串"企业级关键词"组件：OPA 策略引擎、Redis 分布式缓存、
Prometheus 指标、多租户隔离。本项目把它们**全删了**，换成单机真在用的轻量实现。

## 2. 删了什么 → 换成什么（对照表）

| 删掉（第五章"明确不做"）| 换成的轻量实现 | 代码 |
|---|---|---|
| OPA 策略引擎 `app.policy` | 权限门（风险分级 + HITL）+ 轻量输入/输出护栏 | `src/agent/permission_gate.py`、`src/input_sanitizer.py` |
| Redis `app.redis_client` | 进程内 LRU 检索缓存（reindex 后 `cache_clear`）| `src/cache.py` |
| Prometheus `app.metrics` | 结构化日志 + JSONL 链路 trace | `src/logging_utils.py`、`src/trace.py`（见 08）|
| 多租户 / 工单 / customer 表 | 删表；`tenant_id` 只作**检索隔离维度**保留 | 删 `agent/tools.py`、`/agent/ticket`（见 10）|
| Postgres 业务库 | AgentRun/Step 的 JSONL 落盘 | `src/db/`（见 07）|

## 3. 代码怎么做的（删 + 替代的落点）

- 迁移时把 `app.*` 命名空间统一到 `src.*`，上述模块**不迁**，调用点直接删或改接轻量实现。
- 例：`tool_registry.py` 原来有"Redis 优先 + DB 幂等"两段，删掉后只留**进程内幂等缓存**（且只缓存成功结果，见 D-13）。
- 例：`routes_rag.py` 原来每个端点先跑 `evaluate_policy`（OPA），删掉后只保留 `InputGuard`（轻量护栏）+ `gates.py`（分数门控）。

## 4. 为什么必须这么造（工程约束）

- **个人项目撑不起 = 空壳**：单用户、无跨租户、无分布式、无线上告警需求。留着这些组件，
  一问"多租户怎么隔离的""OPA 策略谁维护""Prometheus 告警规则呢"就答不上来——**空壳比没有更减分**。
- **它们还制造断链**：代码 import 着 `app.policy`/`app.redis_client`/`app.metrics`，但模块没迁过来 →
  一堆 `ModuleNotFoundError`（迁移后实测 18 处断链，见 docs/00 D-27 之前）。删干净才 import 通。
- **判据不是"企业级=好"**，是"**这个组件在本项目有没有真在用、能不能被问到底**"。

## 5. 怎么验证它对

- `python scripts/audit_imports.py`：应报 **0 残留 app.* / 0 断链 / 0 引用已删模块**。
- `grep -rn "app.policy\|app.redis_client\|app.metrics\|behavior_guard\|import OPA" src/` 应为空。
- `ruff check src tests scripts` 全绿、`pytest --collect-only` 无收集错误、44+ 模块可 import。
- 替代实现各自有测试：`test_access_prefilter*`（权限）、`test_agent_harness`（幂等/权限门）。

## 6. 代码位置

- 轻量替代：`src/cache.py`、`src/input_sanitizer.py`、`src/agent/permission_gate.py`、`src/logging_utils.py`、`src/trace.py`、`src/db/`
- 正式账本：`BLUEPRINT.md` 第五章「明确不做」；断链清理记录 `docs/00-决策记录.md`（迁移那轮）
- 同款边界判断：05（不做多 Agent）、07（不用 Postgres）
