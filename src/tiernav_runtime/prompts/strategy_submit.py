"""Submit-phase strategy text for the navigation planner."""

STRATEGY_SUBMIT = (
    "当前阶段：提交。已到达目标附近或预算将尽。\n"
    "GOATBench 导航任务：submit_answer 是确认到达，不是输出答案。\n"
    "只有先 navigate_to_object 到目标并 arrived 后才 submit。\n"
    "若 task_mode=question_answering，基于已观测信息给出 answer。\n"
    "不要在未到达目标时盲目 submit。"
)
