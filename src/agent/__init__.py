"""Agent 层：Harness + 状态 + 工具注册表 + 权限门 + 研究工具。

Harness 的四段闭环（迁移时已存在）：
    规划 `_build_plan` → 执行工具 → 评估 `_evaluate_result` → 改写回环

⚠️ 已知逻辑缺陷（见 docs/11-迁移终检报告.md 第三节）：
    改写回环**没有重新检索**——它复用同一批 observations 只换生成用的 query 文本，
    因此评估结果不会改变，最终会走到 `forced_pass: True`。
    **D5 的 Agentic 闭环必须把回环接回检索**，不能复用这个假回环。
"""
