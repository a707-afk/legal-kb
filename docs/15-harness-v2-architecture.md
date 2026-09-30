# Harness V2 架构设计：状态机 + 条件路由

> 位置：`docs/15-harness-v2-architecture.md`
> 前置阅读：[why/06 自研 Harness 的控制循环](../why/06-为什么自研Harness不用LangChain.md)
> 代码位置：`src/agent/harness.py`（V1 主循环）→ V2 拆分为 `src/agent/harness_v2/` 包

---

## 1. 设计目标

V1 Harness 是一条线性流水线：`规划 → 执行 → 草稿 → 评估 → HITL → 落盘`。它的已知缺陷：

- 改写回环复用同一批 observations、不重新检索（假闭环）
- 所有风险等级走同一条路径，低风险问题承担不必要的延迟
- HITL 只有输出审批，无动作级审批
- 预算只有一档（`DEFAULT_BUDGET`），无法按路由差异化
- 审计散落在 `_add_step` 各处，字段不统一
- 会话上下文只有"最近 N 轮原文"，无惰性摘要

V2 的 6 条目标：

1. **状态机驱动**：用显式枚举 + 条件路由替代线性 `for step in plan`，每条路径可独立调试和评测。
2. **三路分流**：按 `risk_level × intent` 把请求分到 FAST_PATH / FULL_PIPELINE / HITL_PRE，低风险 < 30 s 出结果。
3. **HITL 拆分**：动作审批（工具执行前）与输出审批（草稿生成后）分开，高风险动作在执行前就拦住。
4. **护栏闭环**：DRAFT → OUTPUT_GUARD → EVALUATE → REWRITE 形成真正回接检索的循环。
5. **全程审计**：每个状态转移写一条 `AuditEvent`，字段标准化，可回放可聚合。
6. **预算分档**：三档 `RouteBudget`（FAST / FULL / HITL），多维度约束 + 超限降级而非硬中断。

---

## 2. 状态图

### 2.1 主流程

```
┌──────┐     ┌────────────────┐     ┌──────────────┐     ┌────────────┐     ┌───────┐
│ INIT │────▶│ SECURITY_CHECK │────▶│ CONTEXT_LOAD │────▶│ RISK_INTENT│────▶│ ROUTE │
└──────┘     └────────────────┘     └──────────────┘     └────────────┘     └───┬───┘
                   │                                                             │
                   │ (blocked)                          ┌────────────────────────┼────────────────────┐
                   ▼                                    ▼                        ▼                    ▼
              ┌──────────┐                      ┌───────────┐           ┌───────────────┐     ┌───────────┐
              │ REJECTED │                      │ FAST_PATH │           │ FULL_PIPELINE │     │ HITL_PRE  │
              └──────────┘                      └─────┬─────┘           └───────┬───────┘     └─────┬─────┘
                                                      │                         │                   │
                                                      ▼                         ▼                   ▼
                                                ┌───────────┐           ┌───────────┐        ┌───────────┐
                                                │   DRAFT   │           │   PLAN    │        │   PLAN    │
                                                └─────┬─────┘           └─────┬─────┘        └─────┬─────┘
                                                      │                       ▼                    ▼
                                                      │               ┌───────────────┐     ┌──────────────┐
                                                      │               │ PLAN_VALIDATE │     │PLAN_VALIDATE │
                                                      │               └───────┬───────┘     └──────┬───────┘
                                                      │                       ▼                    ▼
                                                      │               ┌───────────────┐     ┌──────────────┐
                                                      │               │ EXECUTE_LOOP  │     │ EXECUTE_LOOP │
                                                      │               └───────┬───────┘     └──────┬───────┘
                                                      │                       │                    │
                                                      ▼                       ▼                    ▼
                                                ┌─────────────────────────────────────────────────────────┐
                                                │                   OUTPUT_GUARD                           │
                                                └────────────────────────────┬────────────────────────────┘
                                                                             │
                                              ┌──────────────────────────────┼──────────────────────┐
                                              ▼                              ▼                      ▼
                                       ┌──────────┐                  ┌───────────┐          ┌───────────┐
                                       │ EVALUATE │                  │   PERSIST │          │ REJECTED  │
                                       └────┬─────┘                  └─────┬─────┘          └───────────┘
                                            │                              ▼
                              (pass) ───────┼─────── (fail)          ┌──────────┐
                              │             │                        │   DONE   │
                              ▼             ▼                        └──────────┘
                        ┌──────────┐  ┌───────────┐
                        │ PERSIST  │  │  REWRITE  │──── (retry < max) ────▶ OUTPUT_GUARD
                        └────┬─────┘  └───────────┘
                             ▼
                        ┌──────────┐
                        │   DONE   │
                        └──────────┘
```

### 2.2 EXECUTE_LOOP 内部循环

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EXECUTE_LOOP                                  │
│                                                                      │
│   ┌───────────┐     ┌──────────────┐     ┌───────────┐     ┌────────────────┐
│   │ TOOL_CALL │────▶│ HITL_ACTION  │────▶│ TOOL_EXEC │────▶│  CHECKPOINT    │
│   │ (选择工具)│     │ (动作审批)   │     │ (执行工具)│     │ (快照当前状态) │
│   └───────────┘     └──────┬───────┘     └───────────┘     └───────┬────────┘
│        ▲                   │                                       │
│        │                   │ (denied)                               │
│        │                   ▼                                       │
│        │            ┌────────────┐                                  │
│        │            │   SKIP /   │                                  │
│        │            │   ABORT    │                                  │
│        │            └────────────┘                                  │
│        │                                                            │
│        └────────────────────────────────────────────────────────────┘
│                         (还有下一步 & 预算未耗尽)
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 护栏闭环

```
DRAFT ──▶ OUTPUT_GUARD ──┬── (pass) ──▶ EVALUATE ──┬── (score ≥ threshold) ──▶ PERSIST
                         │                         │
                         │ (blocked)               │ (score < threshold)
                         ▼                         ▼
                    REJECTED                  REWRITE ──▶ OUTPUT_GUARD (loop)
                                                    │
                                              (exceeds max_rewrite_retries)
                                                    │
                                                    ▼
                                              DEGRADED ──▶ PERSIST (带降级标记)
```

### 2.4 错误处理路径

```
ANY_STATE ── (exception) ──▶ ERROR_CLASSIFY
                                    │
                    ┌───────────────┬┴──────────────┬────────────────┐
                    ▼               ▼               ▼                ▼
              RETRY_BACKOFF    DEGRADED         ABORT → DONE      SKIP → next_step
              (指数退避重试)   (降级出结果)      (终止运行)         (跳过当前步)
                    │
              (exceeds max)
                    │
                    ▼
                DEGRADED
```

---

## 3. 路由逻辑

### 3.1 决策表

| risk_level | intent                    | route         |
|------------|---------------------------|---------------|
| low        | kb_qa / simple_lookup     | FAST_PATH     |
| medium     | kb_qa / analysis / comparison | FULL_PIPELINE |
| high       | any                       | HITL_PRE      |
| critical   | any                       | REJECT or HITL_PRE |

### 3.2 RouteDecision 结构

```python
@dataclass(frozen=True)
class RouteDecision:
    """路由决策结果，由 RISK_INTENT → ROUTE 状态产出。"""
    route: Literal["FAST_PATH", "FULL_PIPELINE", "HITL_PRE", "REJECT"]
    budget: RouteBudget
    reason: str                    # 人类可读的路由原因
    risk_level: str                # low / medium / high / critical
    intent: str                    # kb_qa / simple_lookup / analysis / comparison / ...
    confidence: float              # 意图分类置信度 [0, 1]
```

### 3.3 路由函数签名

```python
def decide_route(risk_level: str, intent: str, intent_confidence: float) -> RouteDecision:
    """纯函数，无副作用。输入风险+意图，输出路由决策+预算。

    规则：
    1. intent_confidence < 0.6 → 升级一档（安全兜底）
    2. critical + 非白名单 intent → REJECT
    3. 查表决定 route，匹配对应 BUDGET_* 预设
    """
```

---

## 4. HITL 拆分设计

V1 只有一个 HITL 点（输出审批）。V2 拆成两个：

| 类型 | 状态 | 触发条件 | 行为 |
|------|------|----------|------|
| 动作审批 | HITL_ACTION | 工具 risk_level ≥ HIGH 且 permission_gate 返回 `requires_approval=True` | 暂停 EXECUTE_LOOP，等人类批准/拒绝该工具调用 |
| 输出审批 | HITL_OUTPUT | 路由为 HITL_PRE，或 EVALUATE 分数 < threshold 且 rewrite 耗尽 | 暂停输出，等人类确认最终答案 |

```python
@dataclass
class HITLRequest:
    """HITL 审批请求（动作或输出）。"""
    request_id: str
    kind: Literal["action", "output"]
    run_id: str
    payload: dict[str, Any]        # action: {tool, params}; output: {draft, citations}
    timeout_seconds: float = 300.0
    created_at: float = field(default_factory=time.time)


@dataclass
class HITLResponse:
    """人类审批结果。"""
    request_id: str
    approved: bool
    reason: str = ""
    modified_payload: dict[str, Any] | None = None  # 人类可修改参数/草稿
```

动作审批被拒后的处理：`SKIP`（跳过该步继续循环）或 `ABORT`（终止整个 run），由 `permission_gate` 返回的 `deny_action` 字段决定。

---

## 5. 会话上下文策略

### 5.1 三层模型

```
Layer 1: 最近 N 轮原文（N=3）
         ↓ 直接注入 system prompt，零 LLM 成本
Layer 2: 第 N+1 ~ M 轮惰性摘要（M=10）
         ↓ 按需生成，结果缓存在 session 对象上
Layer 3: 更早历史
         ↓ 丢弃（不保留，不占预算）
```

### 5.2 惰性摘要触发条件

三条**同时满足**时才生成摘要（否则跳过，只注入 Layer 1）：

1. 会话历史超过 N 轮（`len(session.turns) > 3`）
2. 当前输入含指代标记：`"那个" / "上面" / "刚才" / "继续" / "之前" / "上次"`
3. 当前 run 的 token 预算有余量：`budget.remaining_tokens > summary_cost_estimate`

### 5.3 同步缓存方案

```python
def get_context(session_id: str | None, current_input: str, budget: RouteBudget) -> SessionContext:
    """组装会话上下文（纯同步，摘要结果有缓存）。

    返回 SessionContext:
      - recent_turns: list[Turn]          # Layer 1，最多 N=3 轮原文
      - history_summary: str | None       # Layer 2，惰性摘要（缓存命中则直接返回）
      - summary_cache_key: str | None     # 缓存键 = hash(turns[N:M])
    """
```

摘要缓存策略：以 `hash(turns[N:M])` 为键，会话追加新轮次后键变化 → 自动失效。不引入 Redis，缓存在进程内存 `dict` 中（与现有 `embed_cache` 同策略）。

---

## 6. 全程审计设计

### 6.1 AuditEvent 结构

```python
@dataclass
class AuditEvent:
    """状态机每次转移产出一条审计事件。"""
    event_id: str                       # uuid4
    run_id: str
    timestamp: float                    # time.time()
    from_state: HarnessState
    to_state: HarnessState
    metadata: AuditMetadata


@dataclass
class AuditMetadata:
    """审计元数据，按字段组分类。"""

    # ── 路由决策组 ──
    route: str | None = None
    risk_level: str | None = None
    intent: str | None = None
    intent_confidence: float | None = None

    # ── 预算消耗组 ──
    steps_used: int | None = None
    tool_calls_used: int | None = None
    tokens_used: int | None = None
    latency_ms: float | None = None
    transitions_used: int | None = None
    budget_exhausted: str | None = None      # 哪个维度先耗尽

    # ── 工具调用组 ──
    tool_name: str | None = None
    tool_params_hash: str | None = None
    tool_result_status: str | None = None    # success / error / timeout / denied
    tool_latency_ms: float | None = None

    # ── 护栏组 ──
    guard_verdict: str | None = None         # pass / blocked / degraded
    guard_reason: str | None = None

    # ── 评估组 ──
    eval_score: float | None = None
    eval_dimensions: dict[str, float] | None = None  # {faithfulness, relevance, citation_accuracy}
    rewrite_count: int | None = None

    # ── HITL 组 ──
    hitl_kind: str | None = None             # action / output
    hitl_approved: bool | None = None
    hitl_latency_ms: float | None = None

    # ── 错误组 ──
    error_type: str | None = None
    error_action: str | None = None          # RETRY / DEGRADE / ABORT / SKIP
    retry_count: int | None = None

    # ── 会话组 ──
    session_id: str | None = None
    context_layers_used: list[str] | None = None  # ["layer1", "layer2"]
```

### 6.2 写入点

| 状态转移 | 写入的 metadata 字段组 |
|----------|----------------------|
| INIT → SECURITY_CHECK | （无，安全检查内部记录） |
| SECURITY_CHECK → REJECTED | guard_verdict, guard_reason |
| RISK_INTENT → ROUTE | route, risk_level, intent, intent_confidence |
| ROUTE → FAST_PATH/FULL_PIPELINE/HITL_PRE | route, budget 预设名 |
| EXECUTE_LOOP 内每次 TOOL_CALL | tool_name, tool_params_hash, tool_result_status, tool_latency_ms |
| HITL_ACTION → approved/denied | hitl_kind, hitl_approved, hitl_latency_ms |
| OUTPUT_GUARD → pass/blocked | guard_verdict, guard_reason |
| EVALUATE → pass/fail | eval_score, eval_dimensions, rewrite_count |
| ERROR_CLASSIFY → action | error_type, error_action, retry_count |
| * → DONE | steps_used, tool_calls_used, tokens_used, latency_ms, budget_exhausted |

### 6.3 落盘方式

沿用 V1 的 JSONL 落盘（`data/traces/{date}/`），每条 `AuditEvent` 序列化为一行 JSON。不引入新存储。

---

## 7. 预算控制

### 7.1 RouteBudget 结构

```python
@dataclass(frozen=True)
class RouteBudget:
    """多维度执行预算，路由时确定，运行中只减不增。"""
    max_steps: int                    # 计划最大步数
    max_tool_calls: int               # 工具调用总次数上限
    max_tokens: int                   # LLM token 总消耗上限
    max_rewrite_retries: int          # 改写重试次数
    max_latency_ms: int               # 端到端延迟上限
    max_transitions: int              # 状态转移次数上限（防死循环）
    loop_detect_threshold: int        # 同一观测指纹出现 N 次 → 判定循环
```

### 7.2 三档预设

```python
BUDGET_FAST = RouteBudget(
    max_steps=3,
    max_tool_calls=3,
    max_tokens=4000,
    max_rewrite_retries=0,
    max_latency_ms=30_000,
    max_transitions=10,
    loop_detect_threshold=2,
)

BUDGET_FULL = RouteBudget(
    max_steps=10,
    max_tool_calls=20,
    max_tokens=32_000,
    max_rewrite_retries=2,
    max_latency_ms=120_000,
    max_transitions=30,
    loop_detect_threshold=2,
)

BUDGET_HITL = RouteBudget(
    max_steps=10,
    max_tool_calls=20,
    max_tokens=32_000,
    max_rewrite_retries=1,
    max_latency_ms=300_000,
    max_transitions=40,
    loop_detect_threshold=2,
)
```

### 7.3 超限处理

预算耗尽时不硬中断，而是触发 `ERROR_CLASSIFY` → `DEGRADE`：

- `max_steps` / `max_tool_calls` 耗尽 → 停止规划，用已有 observations 直接进 DRAFT
- `max_tokens` 耗尽 → 截断证据块，降级出结果
- `max_latency_ms` 耗尽 → 立即进入 OUTPUT_GUARD，跳过 EVALUATE
- `max_transitions` 耗尽 → ABORT（兜底防死循环）
- `loop_detect_threshold` 触发 → DEGRADE + 审计标记 `budget_exhausted="loop_detected"`

---

## 8. 错误处理与幂等恢复

### 8.1 ErrorAction 枚举

```python
class ErrorAction(str, Enum):
    RETRY = "retry"        # 指数退避重试（最多 3 次）
    DEGRADE = "degrade"    # 降级出结果（带 degraded=True 标记）
    ABORT = "abort"        # 终止运行，status=failed
    SKIP = "skip"          # 跳过当前步，继续下一步
```

### 8.2 错误分类规则

| 错误类型 | ErrorAction | 说明 |
|----------|-------------|------|
| 工具超时（单次） | RETRY | 指数退避：1s → 2s → 4s |
| 工具超时（重试耗尽） | SKIP | 跳过该步，审计记录 |
| LLM 返回格式异常 | RETRY | 最多 2 次，之后 DEGRADE |
| permission_gate 拒绝 | SKIP or ABORT | 由 deny_action 决定 |
| 预算耗尽 | DEGRADE | 见 §7.3 |
| 未预期异常 | ABORT | 安全兜底 |
| 循环检测触发 | DEGRADE | 用已有证据强制出结果 |

### 8.3 Checkpoint 机制

```python
@dataclass
class Checkpoint:
    """EXECUTE_LOOP 每步完成后写一次快照，用于幂等恢复。"""
    checkpoint_id: str
    run_id: str
    step_index: int
    state_snapshot: dict[str, Any]     # 序列化的关键状态
    observations_so_far: list[dict]    # 已收集的观测
    budget_remaining: RouteBudget      # 剩余预算
    created_at: float
```

恢复逻辑：run 中断后重启时，检查是否有同 `run_id` 的 Checkpoint：
- 有 → 从 `step_index + 1` 继续执行（幂等：相同工具+相同参数跳过）
- 无 → 从头执行

Checkpoint 存储在内存 `dict`（与 JSONL trace 同生命周期），不引入外部存储。

---

## 9. 模块接口定义

### 9.1 HarnessState 枚举

```python
class HarnessState(str, Enum):
    """V2 状态机的全部状态。"""
    INIT = "init"
    SECURITY_CHECK = "security_check"
    CONTEXT_LOAD = "context_load"
    RISK_INTENT = "risk_intent"
    ROUTE = "route"
    FAST_PATH = "fast_path"
    PLAN = "plan"
    PLAN_VALIDATE = "plan_validate"
    EXECUTE_LOOP = "execute_loop"
    DRAFT = "draft"
    OUTPUT_GUARD = "output_guard"
    EVALUATE = "evaluate"
    REWRITE = "rewrite"
    HITL_PRE = "hitl_pre"
    HITL_ACTION = "hitl_action"
    HITL_OUTPUT = "hitl_output"
    ERROR_CLASSIFY = "error_classify"
    RETRY_BACKOFF = "retry_backoff"
    DEGRADED = "degraded"
    CHECKPOINT = "checkpoint"
    PERSIST = "persist"
    DONE = "done"
    REJECTED = "rejected"
```

### 9.2 UserContext 升级

```python
@dataclass
class UserContext:
    """用户上下文（从 V1 的 dict 升级为强类型）。"""
    tenant_id: str
    user_id: str
    roles: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    department: str | None = None
    # V2 新增
    session_id: str | None = None
    risk_override: str | None = None       # 管理员可覆盖风险等级
    approved_tools: list[str] = field(default_factory=list)  # 预批准的工具列表
```

### 9.3 状态转移函数签名

```python
TransitionFn = Callable[["HarnessContext"], tuple[HarnessState, AuditMetadata]]

@dataclass
class HarnessContext:
    """状态机运行时上下文，贯穿整个 run。"""
    run_id: str
    state: HarnessState
    objective: str
    user_context: UserContext
    session_context: SessionContext | None
    route_decision: RouteDecision | None
    budget: RouteBudget
    budget_used: BudgetUsage             # 已消耗量
    plan: list[PlanStep] | None
    observations: list[dict]
    draft: str | None
    eval_result: EvalResult | None
    errors: list[dict]
    audit_log: list[AuditEvent]
    checkpoints: list[Checkpoint]
    transitions: int                     # 已执行转移次数
    started_at: float
```

### 9.4 状态机引擎

```python
class HarnessStateMachine:
    """V2 状态机引擎。

    职责：
    1. 维护 state → transition_fn 的注册表
    2. 执行转移循环直到终态（DONE / REJECTED）
    3. 每次转移写 AuditEvent
    4. 检查预算约束，超限触发 ERROR_CLASSIFY
    """

    def __init__(self, context: HarnessContext):
        self._ctx = context
        self._transitions: dict[HarnessState, TransitionFn] = {}

    def register(self, state: HarnessState, fn: TransitionFn) -> None: ...
    async def run(self) -> AgentHarnessResult: ...
    def _check_budget(self) -> bool: ...
    def _emit_audit(self, to_state: HarnessState, meta: AuditMetadata) -> None: ...
```

---

## 10. permission_gate 调用位置

V2 有两个 permission_gate 调用点：

| # | 调用位置 | 状态 | 检查内容 | 返回处理 |
|---|----------|------|----------|----------|
| 1 | EXECUTE_LOOP 内，工具执行前 | HITL_ACTION | tool.risk_level vs user_context.scopes | `allowed=True` → 继续; `requires_approval=True` → HITL; `allowed=False` → SKIP/ABORT |
| 2 | OUTPUT_GUARD，草稿生成后 | HITL_OUTPUT | route==HITL_PRE 或 eval 不通过 | 强制走输出审批 |

### UserContext 升级对 permission_gate 的影响

```python
# V1 调用方式（不变）
perm = check_permission(tool_def, user_context_dict, params)

# V2：user_context_dict 从 UserContext dataclass 生成
perm = check_permission(tool_def, asdict(ctx.user_context), params)
```

`permission_gate.py` 的 `check_permission` 函数签名**不变**，入参仍是 `dict`。V2 只在调用侧把 `UserContext` 转成 dict 传入。新增字段（`risk_override` / `approved_tools`）由 gate 内部按 `dict.get()` 读取，向后兼容。

---

## 11. 生产级模块清单

### 已有（可直接复用）

| 模块 | 文件 | V2 用途 |
|------|------|---------|
| ToolRegistry | `src/agent/tool_registry.py` | EXECUTE_LOOP 工具执行 |
| PermissionGate | `src/agent/permission_gate.py` | HITL_ACTION 权限检查 |
| OutputGuard | `src/gates.py` | OUTPUT_GUARD 状态 |
| SessionMgr | `src/session_mgr.py` | CONTEXT_LOAD 会话历史 |
| Trace/Audit | `src/trace.py` + JSONL 落盘 | AuditEvent 持久化 |
| LLM Client | `src/llm.py` | PLAN / DRAFT / EVALUATE / REWRITE |
| Embeddings | `src/embeddings.py` | 检索工具内部 |
| Retrieval | `src/retrieval.py` | FAST_PATH + EXECUTE_LOOP 检索步 |
| InputSanitizer | `src/input_sanitizer.py` | SECURITY_CHECK 注入检测 |
| Currency | `src/currency.py` | 检索后时效重排 |
| Citation | `src/citation.py` | DRAFT 引用校验 |

### 需新建

| 模块 | 文件（计划） | 职责 |
|------|-------------|------|
| HarnessState 枚举 | `src/agent/harness_v2/states.py` | 状态定义 + 转移注册表 |
| StateMachine 引擎 | `src/agent/harness_v2/engine.py` | 转移循环 + 预算检查 + 审计发射 |
| Router | `src/agent/harness_v2/router.py` | `decide_route()` 纯函数 |
| Budget | `src/agent/harness_v2/budget.py` | RouteBudget 定义 + 三档预设 + BudgetUsage |
| Audit | `src/agent/harness_v2/audit.py` | AuditEvent / AuditMetadata 定义 + 序列化 |
| Checkpoint | `src/agent/harness_v2/checkpoint.py` | Checkpoint 管理 + 恢复逻辑 |
| Context Builder | `src/agent/harness_v2/context.py` | 三层会话上下文组装 + 惰性摘要 |
| Error Classifier | `src/agent/harness_v2/errors.py` | ErrorAction 枚举 + 分类规则 |
| HITL Manager | `src/agent/harness_v2/hitl.py` | HITLRequest/Response + 超时处理 |
| Transitions | `src/agent/harness_v2/transitions/` | 每个状态的具体转移函数（一文件一状态或按组） |
| V2 入口 | `src/agent/harness_v2/__init__.py` | `run_agent_harness_v2()` — 签名与 V1 返回值兼容 |

---

## 12. 分阶段实施计划

### Phase 1：骨架（状态机 + 路由 + 预算）

- 建 `src/agent/harness_v2/` 包结构
- 实现 `HarnessState` 枚举、`HarnessStateMachine` 引擎
- 实现 `decide_route()` + 三档 `RouteBudget`
- 实现主流程转移：INIT → SECURITY_CHECK → CONTEXT_LOAD → RISK_INTENT → ROUTE → {FAST_PATH | PLAN}
- V1 `run_agent_harness()` 保持不动，V2 独立入口 `run_agent_harness_v2()`
- 验收：V2 骨架能跑通 FAST_PATH 路径（mock LLM），现有 186 个测试全部通过

### Phase 2：执行循环 + HITL

- 实现 EXECUTE_LOOP 内部循环（TOOL_CALL → HITL_ACTION → TOOL_EXEC → CHECKPOINT）
- 实现 HITL Manager（动作审批 + 输出审批）
- 接入 permission_gate 两个调用点
- 实现 Checkpoint 写入 + 恢复逻辑
- 验收：FULL_PIPELINE 路径端到端跑通，高风险工具触发 HITL_ACTION

### Phase 3：护栏闭环 + 错误处理

- 实现 DRAFT → OUTPUT_GUARD → EVALUATE → REWRITE 真闭环（REWRITE 重新调 Retrieval）
- 实现 ErrorAction 分类 + RETRY_BACKOFF + DEGRADED
- 实现循环检测（观测指纹 hash 重复判定）
- 验收：改写回环能真正改善评估分数（对比 V1 的 forced_pass）

### Phase 4：审计 + 会话 + 切换

- 实现 AuditEvent 全程写入 + JSONL 落盘
- 实现三层会话上下文 + 惰性摘要
- V2 通过 feature flag 灰度切换（`HARNESS_VERSION=v2`）
- 回归测试：V2 在现有评测集上的指标 ≥ V1
- 验收：审计日志可回放完整决策链路

---

## 13. 约束与不做清单

| 约束 | 说明 |
|------|------|
| 不引入新外部依赖 | 无 Redis / 无消息队列 / 无 LangGraph / 无 Celery |
| AgentHarnessResult 返回值格式不变 | V2 的 `run_agent_harness_v2()` 返回类型与 V1 完全一致 |
| 现有 186 个测试必须继续通过 | V1 入口不删除，V2 是并行新增 |
| permission_gate.py 接口稳定 | `check_permission()` 签名不变，V2 只在调用侧适配 |
| tool_registry.py 接口稳定 | `ToolRegistry.get()` / `.execute()` 签名不变 |
| 多租户隔离 | V2 不做租户级状态机实例隔离，沿用 V1 的 tenant_id 字段 |
| 沙箱执行 | 工具仍在进程内执行，不引入容器沙箱 |
| 分布式状态 | 状态机单进程运行，不做跨进程状态同步 |
| 异步 HITL 回调 | HITL 等待用 `asyncio.Event`，不引入 webhook / WebSocket 推送 |

---

## 附：V1 → V2 对照

| 维度 | V1 | V2 |
|------|----|----|
| 控制流 | 线性 `for step in plan` | 状态机 + 条件路由 |
| 路由 | 无（所有请求同路径） | risk_level × intent → 三路分流 |
| 预算 | 单档 `DEFAULT_BUDGET` | 三档 `RouteBudget`（FAST/FULL/HITL） |
| HITL | 仅输出审批 | 动作审批 + 输出审批 |
| 改写回环 | 假闭环（不重新检索） | 真闭环（REWRITE → Retrieval → OUTPUT_GUARD） |
| 审计 | `_add_step` 散落 | 统一 `AuditEvent` + `AuditMetadata` |
| 错误处理 | try/except + termination_reason | ErrorAction 四分类 + 指数退避 |
| 会话上下文 | 最近 N 轮原文 | 三层模型 + 惰性摘要 |
| 幂等恢复 | 无 | Checkpoint 快照 + 断点续跑 |
| 循环检测 | 声明但未实现 | 观测指纹 hash + threshold 判定 |
