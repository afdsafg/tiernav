"""Navigate-phase strategy text for the navigation planner."""

STRATEGY_NAVIGATE = (
    "当前阶段：导航。goal object 或相关物体已在观测中出现。\n"
    "若 goal object 直接可见，调用 navigate_to_object 接近目标。\n"
    "若仅看到相关物体（非 goal 本身），navigate 过去以获取更多信息，到达后重新观察。\n"
    "到达目标后验证观测与 goal 一致。若目标丢失，回退到 explore 阶段。\n"
    "GOATBench 导航任务：成功条件是 agent 物理到达目标 1m 内。\n"
    "看到目标不等于成功，必须 navigate_to_object 到目标附近才算到达。\n"
    "到达后才可 submit_answer 确认。不要在远处看到目标就直接 submit。"
)
