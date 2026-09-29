# 08 检索链路 trace + 回放（查询·可观测这一环）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：贯穿查询流程每一步。
> 这篇讲**trace 实际记了什么、怎么落盘、怎么回放，以及为什么不是 Prometheus**。

## 1. 它在系统里是什么

- 每次查询开一个 `TraceRecorder`，检索每经过一个 stage（route/vector/bm25/parallel/fuse/rerank/currency/finish）
  就记一行。事后能按 `query_id` 把整条链路回放出来，定位"这一步掉在哪"。
- 它是**排障/失败归因**工具，不是聚合监控。

## 2. 数据长什么样（落盘布局）

```
data/traces/<YYYY-MM-DD>/<query_id>.jsonl     # 一个 query 一个文件（回放 O(1)，不扫全量）
```
一行一个 stage：
```json
{"ts":"...","query_id":"q-20260928-212942-237cb9","stage":"parallel","elapsed_ms":142.1,
 "vector":20,"bm25":20,"timeout_s":20.0,"degraded":null}
```
候选只落 `file / heading_path / status_code / score`，**不落全文**（避免 trace 泄敏感内容）。

## 3. 代码怎么做的（`src/trace.py`）

- `TraceRecorder.record(stage, **fields)`：追加一行 JSON，带 `elapsed_ms`；**写失败只记日志、绝不抛**（trace 不能拖垮检索）。
- `stage_candidates(stage, nodes)`：记候选数 + `_brief(nodes)`（只留 file/heading/status/score）。
- `create_trace()`：按 `retrieval_trace_enabled` 决定是真记还是 no-op。
- `load_trace(query_id)` / `list_recent_traces()`：回放/列举。
- 埋点在 `src/retrieval.py`：`_tr("route", ...)` / `trace.stage_candidates("vector"/"bm25"/"fuse"/"rerank", ...)` /
  `trace.record("parallel", ..., degraded=...)`；`src/telemetry.py::trace_span` 是计时 span 的接入点。
- 回放脚本 `scripts/replay_trace.py`（`--last` / `--list` / `<query_id>`）。

## 4. 为什么必须这么造（工程约束）

- **Prometheus 看不见 RAG 的四种失败**：没召回 / 排序错 / 用错片段 / 引用错条号——
  这些 HTTP 200、延迟正常，聚合指标一片绿，你却不知道错在哪。只有**按 query 的链路 trace** 能定位。
- **一 query 一文件**：回放是 O(1)（直接读那个文件），不用扫全量日志。
- **不落全文**：trace 会被人翻看，落全文=泄敏感内容；只落定位需要的元信息。

## 5. 怎么验证它对

```bash
python scripts/smoke_retrieval.py --query "公司拖欠工资能解除劳动合同吗"   # 产一条 trace
python scripts/replay_trace.py --last                                     # 回放它
```
应看到完整链路：route（language/collection/candidate_k）→ vector/bm25（各 count）→
parallel（是否 degraded）→ fuse（fusion/count）→ rerank（reranked_from）→ finish（total_ms）。

## 6. 代码位置

- `src/trace.py`（`TraceRecorder` / `create_trace` / `load_trace` / `list_recent_traces`）
- `src/telemetry.py`（`trace_span`）、`src/retrieval.py`（各 stage 埋点）、`scripts/replay_trace.py`
- 正式账本：`BLUEPRINT.md` D-15
