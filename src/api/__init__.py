"""API 层。

    routes_rag    /retrieve  /chat（含 SSE）
    routes_agent  /agent/run

依赖已裁剪完毕：input_sanitizer 重写为轻量本地版（InputGuard / OutputGuard）；
OPA 策略引擎按 BLUEPRINT 第五章「明确不做」已移除全部调用点；
一代客服的 /agent/ticket（customer/ticket）已删除，任务单形态见 D-11（D5 重建）。
"""
