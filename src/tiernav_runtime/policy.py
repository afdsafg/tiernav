"""Pure functional workflow routing policy for the TierNav runtime."""
from __future__ import annotations

from pydantic import Field

from .contracts import EpisodeState, RuntimeModel, RunSpec


class PolicyDecision(RuntimeModel):
    """Routing decision produced by the workflow policy."""

    route: str
    reason: str
    hint: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class WorkflowPolicy:
    """Decides the runtime's next route from RunSpec and EpisodeState.

    Pure function: no external services, no LangGraph, no fabricated evidence.
    """

    def decide(self, spec: RunSpec, state: EpisodeState) -> PolicyDecision:
        # Order matters: stall recovery is checked first per the redesign plan.
        if spec.ablation.stall_recovery and state.failure_type == "stalled":
            return PolicyDecision(route="recover_stall", reason="stalled")

        if state.current_decision is not None and (
            state.current_decision.action_type == "submit_answer"
        ):
            return PolicyDecision(route="finalize", reason="submit_answer")

        if state.round_index >= spec.max_rounds:
            return PolicyDecision(route="fallback", reason="round_budget")

        if state.step_index >= spec.max_steps:
            return PolicyDecision(route="fallback", reason="step_budget")

        return PolicyDecision(route="execute_tool", reason="continue")
