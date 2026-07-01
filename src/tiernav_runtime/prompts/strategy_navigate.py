"""Navigate-phase strategy text for the navigation planner."""

STRATEGY_NAVIGATE = (
    "当前阶段：导航。goal object 或相关物体已在观测中出现。\n"
    "若 goal object 直接可见，调用 navigate_to_object 接近目标。\n"
    "若仅看到相关物体（非 goal 本身），navigate 过去以获取更多信息，到达后重新观察。\n"
    "到达后验证观测与 goal 一致。若目标丢失，回退到 explore 阶段。"
)
