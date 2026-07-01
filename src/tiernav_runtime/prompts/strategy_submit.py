"""Submit-phase strategy text for the navigation planner."""

STRATEGY_SUBMIT = (
    "当前阶段：提交。已接近目标或预算将尽。\n"
    "验证当前观测与 goal 一致后 submit_answer。\n"
    "若 task_mode=question_answering，基于已观测信息给出答案。\n"
    "不要在未确认目标时盲目 submit。"
)
