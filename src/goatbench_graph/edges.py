"""Conditional edges for the GOATBench graph.

- `after_vlm_decide`: on VLM/set_next_navigation_point failure (terminal=True),
  skip navigate and go straight to check_arrival for scoring + END. Otherwise
  proceed to navigate.
- `after_check_arrival`: routes back to observe ("continue") when the subtask
  is not finished and step budget remains, or to END ("end") when terminal
  (success, failure, or step budget exhausted).
"""
from __future__ import annotations


def after_vlm_decide(state: dict) -> str:
    """Conditional edge leaving vlm_decide_node.

    Returns:
      "check_arrival" → check_arrival (VLM failed, terminal; score + end)
      "navigate"      → navigate (normal flow)
    """
    if state.get("terminal", False):
        return "check_arrival"
    return "navigate"


def after_check_arrival(state: dict) -> str:
    """Conditional edge leaving check_arrival_node.

    Returns:
      "continue" → observe (another step)
      "end"      → END (success, failure, or step budget exhausted)
    """
    if state.get("terminal", False):
        return "end"
    return "continue"
