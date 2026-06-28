"""Planner adapter bridging legacy PlannerAction to runtime contracts."""
from __future__ import annotations

from typing import Any

from .contracts import PlannerDecision

# Fields collected into PlannerDecision.arguments when non-None on the
# source action. Kept explicit so the adapter is stable across PlannerAction
# refactors.
_ARGUMENT_FIELDS = (
    "snapshot_id",
    "object_name",
    "seed_id",
    "frontier_id",
    "view_idx",
    "answer",
)


def planner_action_to_decision(action: Any) -> PlannerDecision:
    """Convert a legacy PlannerAction into a validated PlannerDecision.

    Drops None optional fields and lets PlannerDecision clamp confidence.
    """
    confidence = getattr(action, "confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    arguments = {
        field: getattr(action, field, None)
        for field in _ARGUMENT_FIELDS
        if getattr(action, field, None) is not None
    }

    return PlannerDecision(
        action_type=getattr(action, "action_type", ""),
        reasoning=getattr(action, "reason", "") or "",
        expected=getattr(action, "expected", "") or "",
        confidence=confidence,
        arguments=arguments,
    )
