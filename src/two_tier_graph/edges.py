"""Conditional edge functions for the Two-Tier graph.

Two conditional edges:
  - `after_guard` (leaves loop_guard_node): routes submit_answer to submit_node,
    everything else to executor_node. Mirrors agent_workflow.py:1614.
  - `after_memory` (leaves memory_update_node): encodes the for-loop + entity-
    exhaustion + step-budget semantics of agent_workflow.py:1665-1677.

`after_memory` is the single most behavior-sensitive part of the graph. The
ordering MUST be:
  1. round-budget (for-loop end)  → fallback_submit
  2. exhausted_flag (continue)    → continue   (skips step-budget check)
  3. step-budget (break)          → fallback_submit
  4. else                         → continue

Getting this order wrong breaks the `:1665-1677` skip semantics and drifts
benchmark numbers.
"""
from __future__ import annotations

from typing import Any


def after_guard(state: dict) -> str:
    """Conditional edge leaving loop_guard_node.

    Mirrors `if action.action_type == "submit_answer": return result` at
    agent_workflow.py:1614. Returns the name of the next node.
    """
    current_action: Any = state.get("current_action")
    if current_action is not None and current_action.action_type == "submit_answer":
        return "submit"
    return "execute"


def after_memory(state: dict) -> str:
    """Conditional edge leaving memory_update_node.

    Reproduces the for-loop + entity-exhaustion + step-budget semantics of
    agent_workflow.py:1665-1677. Order matters:

      - Round-budget (for-loop end, `:1442` range exhausted) → fallback_submit.
        Checked FIRST because the for-loop is outermost. When rounds_used
        reaches max_planner_rounds, the for-loop exits to the fallback at :1681.
      - exhausted_flag (`:1665-1672` `continue`) → continue. This SKIPS the
        step-budget check at :1675 for that round. Must be checked BEFORE
        step-budget.
      - Step-budget (`:1675-1677` `break`) → fallback_submit.
      - Otherwise → continue (next round).
    """
    rounds_used: int = state.get("rounds_used", 0)
    max_planner_rounds: int = state.get("max_planner_rounds", 0)

    # 1. Round-budget (for-loop end) — checked first, dominates everything.
    if rounds_used >= max_planner_rounds:
        return "fallback_submit"

    # 2. Entity exhaustion → continue (skips step-budget check).
    if state.get("exhausted_flag", False):
        return "continue"

    # 3. Step-budget break.
    steps_taken: int = state.get("steps_taken", 0)
    max_total_steps: int = state.get("max_total_steps", 0)
    if steps_taken >= max_total_steps:
        return "fallback_submit"

    # 4. Default: next round.
    return "continue"
