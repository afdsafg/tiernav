"""Explore-phase strategy text for the navigation planner."""

STRATEGY_EXPLORE = (
    "当前阶段：探索。goal object 尚未直接可见。\n"
    "优先 explore_panorama 观察环境，再 explore_frontier 扩展未知区域。\n"
    "每轮换一个未访问 frontier，勿重复。注意 available_targets 中的未访问 room。\n"
    "若看到 goal object 本身，或看到与 goal 语义相关的物体（如找冰箱时看到烤箱/灶台/微波炉等厨房用品），切换到 navigate 阶段。"
)
