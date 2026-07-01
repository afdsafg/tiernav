"""Externalized prompt strategy text for the TierNav runtime.

The skeleton (identity, output format, tool rules, examples) is task-mode
specific. Phase-specific guidance lives in ``strategy_{explore,navigate,submit}.py``
and is selected at compile time via :func:`strategy_for_phase`.
"""
from __future__ import annotations

from ..contracts import TaskMode
from .strategy_explore import STRATEGY_EXPLORE
from .strategy_navigate import STRATEGY_NAVIGATE
from .strategy_submit import STRATEGY_SUBMIT

# Shared tool-availability rules (identical for both task modes).
_TOOL_RULES = (
    "Pick frontier_id / seed_id / object_name from the available_targets section below. Do NOT invent ids.\n"
    "Do not call explore_frontier when frontiers is none or absent.\n"
    "Do not call explore_seed when seeds is none or absent.\n"
    "Do not call navigate_to_object when objects is none or absent.\n"
    "For target tools, copy the exact frontier_id, seed_id, or object_name from available_targets.\n"
    "query_scene_memory: call ONLY when you believe relevant goal/answer info exists in memory but is not shown in the current prompt. Recalled content persists across rounds — do not query the same content twice."
)

STRATEGY_SKELETON = (
    "You are a navigation planner. Output ONLY a JSON object on a single line, no markdown fences, no prose.\n"
    "Required fields: action_type (one of the available tools), reason (string), expected (string).\n"
    "Optional fields: object_name (str), seed_id (str), frontier_id (str), view_idx (int), answer (str, only for question_answering submit).\n"
    + _TOOL_RULES + "\n"
    "Strategy: explore_panorama to observe -> explore_frontier/explore_seed to move -> navigate_to_object once target visible -> submit_answer after arrived at target.\n"
    "GOATBench navigation: success = agent physically within 1m of goal. Seeing the target is NOT enough — you MUST navigate_to_object to reach it before submit_answer.\n"
    'Example: {"action_type": "explore_panorama", "reason": "Need to observe surroundings", "expected": "Get room layout"}\n'
    'Example: {"action_type": "submit_answer", "reason": "Arrived at refrigerator", "expected": "Goal reached"}'
)

STRATEGY_SKELETON_QA = (
    "You are a question-answering planner. Output ONLY a JSON object on a single line, no markdown fences, no prose.\n"
    "Required fields: action_type (one of the available tools), reason (string), expected (string).\n"
    "Optional fields: object_name (str), seed_id (str), frontier_id (str), view_idx (int), answer (str, required for submit_answer).\n"
    + _TOOL_RULES + "\n"
    "Strategy: explore_panorama to observe -> explore_frontier/explore_seed to explore rooms -> submit_answer with answer when you have enough visual information.\n"
    "AEQA question-answering: success = correct answer. You do NOT need to physically reach any object. Explore to gather visual information about the question, then submit_answer with your answer.\n"
    'Example: {"action_type": "explore_panorama", "reason": "Need to observe surroundings", "expected": "Get room layout"}\n'
    'Example: {"action_type": "submit_answer", "reason": "Saw painting above table", "expected": "Answer question", "answer": "A painting"}'
)

STRATEGY_EXPLORE_QA = (
    "当前阶段：探索。需要收集与问题相关的视觉信息。\n"
    "优先 explore_panorama 观察环境，再 explore_frontier 扩展未知区域。\n"
    "每轮换一个未访问 frontier，勿重复。注意 available_targets 中的未访问 room。\n"
    "寻找与问题语义相关的物体（如问“桌上有什么”时关注桌子及桌上物品），收集足够信息后 submit_answer 给出答案。\n"
    "不需要靠近目标物体，只要能观察到并回答问题即可。"
)

_STRATEGIES = {
    "explore": STRATEGY_EXPLORE,
    "navigate": STRATEGY_NAVIGATE,
    "submit": STRATEGY_SUBMIT,
}

_STRATEGIES_QA = {
    "explore": STRATEGY_EXPLORE_QA,
    "navigate": "",  # AEQA never enters navigate phase
    "submit": STRATEGY_SUBMIT,
}


def strategy_for_phase(phase: str) -> str:
    """Return the GOATBench strategy text for the given phase. Empty string for unknown."""
    return _STRATEGIES.get(phase, "")


def skeleton_for_task(task_mode: TaskMode | str) -> str:
    """Return the prompt skeleton appropriate for the task mode."""
    val = task_mode.value if isinstance(task_mode, TaskMode) else str(task_mode)
    if val == TaskMode.QUESTION_ANSWERING.value:
        return STRATEGY_SKELETON_QA
    return STRATEGY_SKELETON


def strategy_for_phase_task(phase: str, task_mode: TaskMode | str) -> str:
    """Return phase strategy text appropriate for the task mode. Empty string for unknown."""
    val = task_mode.value if isinstance(task_mode, TaskMode) else str(task_mode)
    table = _STRATEGIES_QA if val == TaskMode.QUESTION_ANSWERING.value else _STRATEGIES
    return table.get(phase, "")


# Backwards-compat alias. Phase-3 callers should use STRATEGY_SKELETON +
# strategy_for_phase(phase). Kept so any external import still resolves.
STRATEGY_TEXT = STRATEGY_SKELETON
