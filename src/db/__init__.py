"""持久化层：JSONL 落盘（替代旧 SQLAlchemy/Postgres）。

决策依据 BLUEPRINT D-12 附：个人项目不引入数据库，改为「一个 run 一个文件、
步骤逐条追加」的 JSONL。按 run_id 分文件，天然无写竞争；可直接 diff 与回放。
"""
