# 07 AgentRun/Step 用 JSONL 落盘（Agent·持久化这一环）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：主线三末尾——harness 跑完把 run 和每步落盘。
> 这篇讲**持久化层实际怎么实现的、为什么不是 Postgres**。

## 1. 它在系统里是什么

- Harness 每跑一次，产出一个 `AgentRun`（头）+ 若干 `AgentStep`（每步）。
- 这些要落盘，供事后回放/审计。本项目**不用数据库**，用 JSONL 文件。
- 审计日志（权限允许/拒绝、工具调用）也走同一套 JSONL 机制。

## 2. 数据长什么样（落盘布局）

```
data/agent_runs/<run_id>.jsonl     # 一个 run 一个文件；首行 run 头，后续每行一个 step
data/audit/policy_audit_log.jsonl  # 权限/审计事件，追加
data/audit/tool_call.jsonl         # 工具调用审计，追加
```
每行是一条 JSON（`AgentRun.to_dict()` 带 `"_type":"run"`，`AgentStep.to_dict()` 带 `"_type":"step"`）。

## 3. 代码怎么做的（`src/db/`）

- `src/db/engine.py`：
  - `get_sessionmaker()` 返回 `JsonlSession` 类；`JsonlSession()` 是异步上下文管理器，
    **保留旧的 `async with get_sessionmaker()() as session: session.add(obj); await session.commit()` 调用形态**
    （所以 harness 代码几乎没改，只换了 import）。
  - `commit()` 按 `run_id` 分文件写：run 头在前、step 在后。
  - `append_jsonl()` 是 fire-and-forget：写失败只记日志、**绝不抛**（持久化不能拖垮业务）。
- `src/db/models/agent_run.py` / `agent_step.py`：`@dataclass`（不是 ORM）+ `to_dict()`。
- `src/audit_service.py`：`write_audit_log` / `write_tool_call_audit` 同样 `append_jsonl` 到 `data/audit/`。

## 4. 为什么必须这么造（工程约束）

- **没有业务库需求**：检索型项目不存业务实体（旧项目的多租户/工单/customer 表都删了，见 14）。
  唯一要落的是"这次跑了哪些步"——这是**追加型日志**，不是关系数据，用不上 SQL 的 JOIN/事务/并发控制。
- **按 run_id 分文件天然无写竞争**：不用处理并发写热点。
- **可 diff、可回放**：JSONL 是纯文本，能直接看、直接 grep、直接喂给回放脚本。
- **少一个服务 = 少一个挂点**：不用起 Postgres、不用连接池/迁移。

## 5. 怎么验证它对

- `pytest tests/agent/test_agent_harness.py::TestHarnessRunOffline::test_run_persists_jsonl`：
  跑一条 run（LLM mock 掉），断言 `data/agent_runs/`（测试里指向 tmp）下**至少落一个 .jsonl**。
- 手工：跑一次 `/agent/run` 后 `ls data/agent_runs/`，`cat` 那个文件应看到首行 run、后续 step。

## 6. 代码位置

- `src/db/engine.py`（`get_sessionmaker` / `JsonlSession` / `append_jsonl` / `run_file`）
- `src/db/models/agent_run.py`、`agent_step.py`；`src/audit_service.py`
- 正式账本：`BLUEPRINT.md` D-12 附；同款取舍见 `docker-compose.yml` 注释「不用 Postgres/Redis/worker」
