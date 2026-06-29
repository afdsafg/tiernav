"""Benchmark-specific success evaluation for the TierNav runtime.

The runtime does not judge answer quality — that is an external LLM-Match
step for AEQA. This evaluator produces only the *runtime* success verdict:

* AEQA (``QUESTION_ANSWERING``): success iff the planner explicitly
  submitted a non-empty answer. Distance is irrelevant.
* GOATBench (``GOAL_NAVIGATION``): success iff the planner explicitly
  submitted (signaled stop) AND the final agent-to-goal distance is within
  ``rule.success_distance_m``. Snapshot presence alone never implies
  success — the evaluator takes no snapshot argument by design.

Task 7 wires this evaluator into ``finalize_node``; until then it is a
standalone module.
"""
from __future__ import annotations

from .contracts import BenchmarkRule, RuntimeModel, TaskMode


class SuccessVerdict(RuntimeModel):
    """Runtime success verdict with a machine-readable reason."""

    success: bool
    reason: str


class SuccessEvaluator:
    """Produces a runtime success verdict from a :class:`BenchmarkRule`.

    Construct once per benchmark rule; call :meth:`evaluate` per episode.
    Distance is passed in (not computed here) because measuring it requires
    the scene/executor, which lives in the environment adapter (Task 8).
    """

    __slots__ = ("_rule",)

    def __init__(self, rule: BenchmarkRule) -> None:
        self._rule = rule

    def evaluate(
        self,
        task_mode: TaskMode,
        *,
        submitted_explicitly: bool = False,
        answer: str = "",
        distance_to_goal: float | None = None,
    ) -> SuccessVerdict:
        if task_mode is TaskMode.QUESTION_ANSWERING:
            return self._evaluate_aeqa(submitted_explicitly, answer)
        if task_mode is TaskMode.GOAL_NAVIGATION:
            return self._evaluate_goatbench(submitted_explicitly, distance_to_goal)
        raise ValueError(f"unsupported task_mode: {task_mode!r}")

    # --- AEQA --------------------------------------------------------------

    @staticmethod
    def _evaluate_aeqa(submitted_explicitly: bool, answer: str) -> SuccessVerdict:
        if not submitted_explicitly:
            return SuccessVerdict(success=False, reason="no_explicit_submit")
        if not answer:
            return SuccessVerdict(success=False, reason="no_answer")
        return SuccessVerdict(success=True, reason="answer_submitted")

    # --- GOATBench ---------------------------------------------------------

    def _evaluate_goatbench(
        self, submitted_explicitly: bool, distance_to_goal: float | None
    ) -> SuccessVerdict:
        # Explicit-stop requirement is enforced before distance: a budget
        # fallback that happens to be near the goal is not a success.
        if self._rule.requires_explicit_stop and not submitted_explicitly:
            return SuccessVerdict(success=False, reason="no_explicit_submit")
        if distance_to_goal is None:
            return SuccessVerdict(success=False, reason="no_distance")
        if distance_to_goal <= self._rule.success_distance_m:
            return SuccessVerdict(success=True, reason="distance_within_threshold")
        return SuccessVerdict(success=False, reason="distance_exceeded")
