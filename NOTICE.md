# 数据来源与授权声明

本项目使用以下公开数据集。所有数据均来自公开渠道，授权情况如下。

---

## 1. 国家法律法规数据库（法规 / 行政法规 / 司法解释）

- **来源**：https://flk.npc.gov.cn
- **内容**：法律 734 条 / 行政法规 851 条 / 司法解释 881 条（含历史版本），官方 Word 原件
- **性质**：政府公开信息
- **授权**：可自由使用
- **获取方式**：通过 `law-crawler-unified`（MIT）批量下载

## 2. LeCaRD（中文法律案例检索数据集）

- **来源**：https://github.com/myx666/LeCaRD（清华大学 THUIR）
- **内容**：107 个查询案例 + 10,718 篇候选刑事判决书 + 学术相关性标注
- **授权**：**MIT License**（© 2021 myx666 / Yixiao Ma）
- **引用**：
  ```
  Yixiao Ma, et al. LeCaRD: A Legal Case Retrieval Dataset for Chinese Law System.
  SIGIR 2021.
  ```
- ⚠️ **本项目第一轮不使用该数据集**（当事人个人信息未脱敏 + 评测存在答案泄漏）。
  原始数据仅本地保留，不入版本库。详见 `docs/04-风险与合规.md`

## 3. LawBench

- **来源**：https://github.com/open-compass/LawBench
- **内容**：任务 1-1 法条内容问答 500 条（含标准答案）
- **授权**：开源（Apache-2.0）
- **引用**：
  ```
  Zhiwei Fei, et al. LawBench: Benchmarking Legal Knowledge of Large Language Models. 2023.
  ```

## 4. 爬虫工具

- **law-crawler-unified**：MIT License

---

## 免责声明

本项目为**个人技术作品**，用于展示检索系统与评测方法的设计能力。

- 系统输出**不构成法律意见**，不得用于实际法律决策
- 法条时效判定依据 flk 官方元数据字段，**不保证与最新立法状态完全同步**
- 引用的法条内容以官方发布为准

---

## 模型

| 模型 | 授权 | 用途 |
|---|---|---|
| `BAAI/bge-m3` | MIT | 嵌入 |
| `BAAI/bge-reranker-v2-m3` | MIT | 重排 |
| Qwen3（Ollama） | Apache-2.0 | 生成 |
