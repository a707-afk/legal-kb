# 12 Qdrant 用 server 模式（建库/查询·向量库这一环）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：建库 [4] 写入 + 查询第 2 步向量检索，都连的是 Qdrant。
> 这篇讲**同一个 Qdrant 有两种接法（嵌入式 path / 独立 server），本项目为什么必须用 server**。

## 1. 它在系统里是什么

- `qdrant-client` 有两种模式：
  - **local（path）模式**：`QdrantClient(path="data/qdrant_local")`，纯 Python 嵌入式，无需起服务。
  - **server 模式**：`QdrantClient(url="http://localhost:18333")`，连一个独立的 Qdrant 进程（Docker）。
- 本项目 88,681 个点，**必须用 server 模式**。

## 2. 怎么切换（配置/接口）

`.env` 两个开关，代码判据在 `src/vector_store.py::_qdrant_client`：
```
qdrant_path 非空 且 qdrant_url 为空  → local 模式
否则                                  → server 模式（url）
```
当前 `.env`：`QDRANT_URL=http://localhost:18333`、`QDRANT_PATH=`（空）→ server 模式。
起服务：`docker compose up -d qdrant`（容器 `legal-kb-qdrant`，映射 127.0.0.1:18333→6333）。

## 3. 代码怎么做的

- `_qdrant_client(settings)`：**复用同一个 client**（local 模式不支持多实例并发打开同一目录），按上面的判据选模式。
- `rebuild_index()`：先 `delete_collection` 再写（⚠️ 见下方坑），`assert_collection_dimension` 核对集合维度==模型维度。
- `get_vector_index()`：加载时也 `assert_collection_dimension`（维度不一致直接抛，不静默返回错结果）。

## 4. 为什么必须这么造（工程约束，都是实测踩到的）

local 模式在 8.8 万点下有三个硬伤：
- **冷向量搜索 ~20s**：local 没有 HNSW，近乎暴力扫；首查直接 40s+。server 走 HNSW，毫秒级。
- **payload 索引无效**：local 建 payload 索引会警告 "no effect"——而 **D2.6 权限预筛（按 tenant/密级过滤）靠 payload 过滤**，local 下只能全扫。
- **官方明确不推荐 >2 万点**；建库写入也慢（local ~500s vs server ~130–200s）。

## 5. 怎么验证它对 + 一个踩过的坑

- `reports/build_index_report.json` 的 `qdrant_mode` 应为 `url`、`collection_points` 应 **== 节点数 88681**。
- ⚠️ **坑（docs/00 D-27）**：local 模式下若上个进程没干净退出（`QdrantClient.__del__` 在解释器关闭时抛
  `ImportError: sys.meta_path is None`），下次 `delete_collection` 会**静默失败**（被 `except: pass` 吞），
  新点追加到旧集合上 → **点数翻倍（177362=2×88681）**。所以验收必须查 **点数 == 节点数**，不能只看"脚本没报错"。
  修法：无进程占锁时显式 `delete_collection` + 验证 `get_collections()==[]` 再重建；脚本末尾显式 `clear_index_memory_cache()` 关客户端。

## 6. 代码位置

- `src/vector_store.py`（`_qdrant_client` / `rebuild_index` / `assert_collection_dimension` / `clear_index_memory_cache`）
- `.env`（`QDRANT_URL` / `QDRANT_PATH`）、`docker-compose.yml`；正式账本 `docs/00-决策记录.md` D-27 / D-28
- 代价：server 模式**依赖 Docker**；连 localhost 要设 `NO_PROXY=localhost,127.0.0.1`（否则代理劫持致 502）
