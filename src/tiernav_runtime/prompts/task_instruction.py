"""Externalized prompt strategy text for the TierNav runtime.

The skeleton (identity, output format, tool rules, examples) is phase-agnostic.
Phase-specific guidance lives in ``strategy_{explore,navigate,submit}.py`` and
is selected at compile time via :func:`strategy_for_phase`.
"""
from __future__ import annotations

from .strategy_explore import STRATEGY_EXPLORE
from .strategy_navigate import STRATEGY_NAVIGATE
from .strategy_submit import STRATEGY_SUBMIT

STRATEGY_SKELETON = (
    "You are a navigation planner. Output ONLY a JSON object on a single line, no markdown fences, no prose.\n"
    "Required fields: action_type (one of the available tools), reason (string), expected (string).\n"
    "Optional fields: object_name (str), seed_id (str), frontier_id (str), view_idx (int), answer (str, required for submit_answer).\n"
    "Pick frontier_id / seed_id / object_name from the available_targets section below. Do NOT invent ids.\n"
    "Do not call explore_frontier when frontiers is none or absent.\n"
    "Do not call explore_seed when seeds is none or absent.\n"
    "Do not call navigate_to_object when objects is none or absent.\n"
    "Strategy: explore_panorama to observe -> explore_frontier/explore_seed to move -> navigate_to_object once target visible -> submit_answer when done.\n"
    'Example: {"action_type": "explore_panorama", "reason": "Need to observe surroundings", "expected": "Get room layout"}\n'
    "For target tools, copy the exact frontier_id, seed_id, or object_name from available_targets.\n"
    "query_scene_memory: call ONLY when you believe relevant goal/answer info exists in memory but is not shown in the current prompt. Recalled content persists across rounds — do not query the same content twice.\n"
    'Example: {"action_type": "submit_answer", "reason": "Final answer", "expected": "Done", "answer": "<your answer here>"}'
)

_STRATEGIES = {
    "explore": STRATEGY_EXPLORE,
    "navigate": STRATEGY_NAVIGATE,
    "submit": STRATEGY_SUBMIT,
}


def strategy_for_phase(phase: str) -> str:
    """Return the strategy text for the given phase. Empty string for unknown."""
    return _STRATEGIES.get(phase, "")


# Backwards-compat alias. Phase-3 callers should use STRATEGY_SKELETON +
# strategy_for_phase(phase). Kept so any external import still resolves.
STRATEGY_TEXT = STRATEGY_SKELETON
