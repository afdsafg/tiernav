"""Shared helpers extracted from agent_workflow.py.

Deduplicated in phase-1 tech-debt cleanup (D5). Previously `_NAV_OBJ_INVALID`,
`_is_valid_object_desc`, and `_build_messages` were defined twice in
agent_workflow.py (~905 and ~1735). Moved here verbatim.

`_build_messages` previously closed over `SYSTEM_PROMPT` (a module global in
agent_workflow.py). To avoid a circular import (shared_helpers ← agent_workflow),
`system_prompt` is now an explicit parameter. Callers that previously relied on
the closure must pass it explicitly.
"""
from typing import List

from src.agent_context import ContextManager


# Invalid arguments for navigate_to_object — these are not object descriptions
_NAV_OBJ_INVALID = {
    "", "forward", "backward", "left", "right", "up", "down",
    "explore", "navigate", "search", "look", "go", "move",
    "room", "room 0", "room 1", "room 2", "room 3", "room 4",
    "frontier", "frontier 0", "frontier 1", "frontier 2",
    "yes", "no", "true", "false", "none", "null",
    "the kitchen", "the bathroom", "the bedroom", "the living room",
    "kitchen", "bathroom", "bedroom", "living room",
}


def _is_valid_object_desc(desc: str) -> bool:
    """Check if a string is a valid concrete object description for GroundingDINO.

    Rejects empty strings, directions, room names, and other non-object terms.
    """
    if not desc or not isinstance(desc, str):
        return False
    desc_clean = desc.strip().lower()
    if desc_clean in _NAV_OBJ_INVALID:
        return False
    if len(desc_clean) < 2:
        return False
    # Reject pure numbers (room/frontier IDs)
    try:
        int(desc_clean)
        return False
    except ValueError:
        pass
    return True


def _build_messages(context: ContextManager, system_prompt: str) -> List[dict]:
    """Build the message list for VLM from context manager state.

    `system_prompt` is passed explicitly to avoid a circular import on
    agent_workflow.SYSTEM_PROMPT.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Add stage transition summaries from previous stages
    for transition in context.transitions:
        if transition.from_stage != context.current_stage:
            summary_text = (
                f"[Stage {transition.from_stage}→{transition.to_stage} summary]\n"
                f"{transition.summary}"
            )
            messages.append({"role": "assistant", "content": summary_text})

    # Add current stage messages
    messages.extend(context.stage_messages)

    return messages
