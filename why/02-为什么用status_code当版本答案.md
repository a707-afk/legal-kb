# 02 status_code 作版本答案 + 交叉验证（建库·导入这一环）

> 在 [00 全景图](00-系统怎么跑起来的.md) 的位置：紧接 01。01 讲"每版一个文件"，
> 这篇讲**每个文件怎么带上"我是哪一版、还有没有效"的机器可读标签**，以及怎么证明这标签可信。

## 1. 它在系统里是什么

- **要解决的问题**：判分需要"哪一版现行有效"的标准答案（GT）。人工标 431 个版本要 1–2 周且不可复现。
- **做法**：直接用官网 metadata 的 `status_code` 当 GT，再用"生效日期最大者"**独立交叉验证**它。
- **落在哪**：写进每个 `.md` 的 front-matter（`status_code` / `status_label`），建库时随节点进 Qdrant payload。

## 2. 数据长什么样

`status_code` 取值（`ingest_npc.STATUS_MAP`）：`3=现行有效 / 2=已被修订 / 1=已废止 / 4=尚未生效 / null=未标注`。
front-matter 里长这样：
```yaml
status_code: 4          # 整数或 null，不是字符串
status_label: "尚未生效"
effective_date: "2026-10-01"
```

## 3. 代码怎么做的

- `scripts/ingest_npc.py::assert_metadata_fields`：**缺 `status_code` 键直接 raise**（值可以是 null，键不能没有）。
- `scripts/ingest_npc.py::build_front_matter`：按 `STATUS_MAP` 写 `status_code` + `status_label`。
- `src/chunking.py::_parse_frontmatter`：**把 `status_code` 从文本强制转成 `int`/`None`**。
  这一步是关键——front-matter 是纯文本，`status_code: 3` 读出来是字符串 `"3"`，
  下游 `sc == 3` 会**永远 False 且不报错**（静默失效）。所以只对这个键做强制转型
  （不碰 `version`，因为 bbbs 可能全是数字、转 int 会丢前导零）。
- `scripts/verify_version_gt.py`：交叉验证——`H1` status_code=4 的生效日期是否都 > 今天；
  `H2` 多版本里 status_code=3 是否就是 effective_date 最大者。

## 4. 为什么必须这么造（工程约束）

- **判分要机器可读**：D 类评测（`run_eval.py` 的 `top1_status_code_eq`）直接比 `node.status_code == 3`；
  若 status_code 是字符串或缺失，判分全错但不报错——所以入库层用断言 + 转型把它焊死。
- **不能自己验自己**：用官方 status_code 当 GT，就必须用**另一个独立信号**（生效日期）交叉验证，
  否则"官方字段错了"无从发现。实测 353/360（98.1%）一致，剩 7 个是"已公布未施行"的合法情形。

## 5. 怎么验证它对

- `python scripts/verify_version_gt.py`：应看到 `status_code=4` 的 **9 条全部** effective_date>今天；
  多版本中 status_code=3 即最新生效版 **353 符合 / 少量不一致**（那少量是未生效新版，属正常）。
- 建库后 `reports/build_index_report.json` 的 `nodes_with_status_code` = 节点总数（**88681/88681**）。

## 6. 代码位置

- `scripts/ingest_npc.py`（`STATUS_MAP` / `assert_metadata_fields` / `build_front_matter`）
- `src/chunking.py::_parse_frontmatter`（status_code 强制转型那段注释务必读）
- `scripts/verify_version_gt.py`（H1/H2 交叉验证）
- 正式账本：`BLUEPRINT.md` D-02
