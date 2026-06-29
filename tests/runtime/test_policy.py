"""Tests for the workflow routing policy."""
from __future__ import annotations

import pytest

from src.tiernav_runtime.contracts import (
    AblationConfig,
    EpisodeState,
    PlannerDecision,
    RunSpec,
)
from src.tiernav_runtime.policy import PolicyDecision, WorkflowPolicy


def _spec(stall_recovery: bool = False, max_rounds: int = 10, max_steps: int = 50) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
        seed=0,
        max_rounds=max_rounds,
        max_steps=max_steps,
        ablation=AblationConfig(stall_recovery=stall_recovery),
    )


def _state(
    *,
    action_type: str = "explore_frontier",
    failure_type: str = "",
    round_index: int = 0,
    step_index: int = 0,
) -> EpisodeState:
    return EpisodeState(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="where is the sofa?",
        round_index=round_index,
        step_index=step_index,
        current_decision=PlannerDecision(action_type=action_type),
        failure_type=failure_type,
    )


# --- Plan examples ---------------------------------------------------------


def test_submit_answer_routes_to_finalize():
    decision = WorkflowPolicy().decide(_spec(), _state(action_type="submit_answer"))
    assert decision.route == "finalize"
    assert decision.reason == "submit_answer"


def test_round_budget_routes_to_fallback():
    decision = WorkflowPolicy().decide(
        _spec(max_rounds=3), _state(round_index=3, action_type="explore_frontier")
    )
    assert decision.route == "fallback"
    assert decision.reason == "round_budget"


def test_step_budget_routes_to_fallback():
    decision = WorkflowPolicy().decide(
        _spec(max_steps=5), _state(step_index=5, action_type="explore_frontier")
    )
    assert decision.route == "fallback"
    assert decision.reason == "step_budget"


def test_normal_explore_routes_to_execute_tool():
    decision = WorkflowPolicy().decide(_spec(), _state(action_type="explore_frontier"))
    assert decision.route == "execute_tool"
    assert decision.reason == "continue"


def test_stall_recovery_disabled_does_not_route_recover_stall():
    decision = WorkflowPolicy().decide(
        _spec(stall_recovery=False), _state(failure_type="stalled", action_type="explore_frontier")
    )
    assert decision.route != "recover_stall"


def test_stall_recovery_enabled_routes_recover_stall():
    decision = WorkflowPolicy().decide(
        _spec(stall_recovery=True), _state(failure_type="stalled", action_type="explore_frontier")
    )
    assert decision.route == "recover_stall"
    assert decision.reason == "stalled"


# --- Extra coverage --------------------------------------------------------


def test_stalled_takes_priority_over_submit_answer():
    """Plan checks stalled before submit_answer, so recover_stall wins."""
    decision = WorkflowPolicy().decide(
        _spec(stall_recovery=True),
        _state(failure_type="stalled", action_type="submit_answer"),
    )
    assert decision.route == "recover_stall"
    assert decision.reason == "stalled"


def test_policy_decision_metadata_not_shared_between_instances():
    a = PolicyDecision(route="x", reason="y")
    b = PolicyDecision(route="x", reason="y")
    a.metadata["k"] = "v"
    assert "k" not in b.metadata


def test_policy_decision_metadata_defaults_to_empty():
    d = PolicyDecision(route="x", reason="y")
    assert d.metadata == {}


def test_hint_defaults_to_empty():
    d = PolicyDecision(route="x", reason="y")
    assert d.hint == ""


# --- Consistency with the success evaluator --------------------------------


def test_submit_answer_routes_to_finalize_so_evaluator_sees_explicit_submit():
    """Policy routes submit_answer to finalize; the success evaluator treats
    reaching finalize via submit_answer as an explicit submit. This pins the
    hand-off contract between Task 6 (evaluator) and Task 7 (graph wiring).
    """
    decision = WorkflowPolicy().decide(_spec(), _state(action_type="submit_answer"))
    assert decision.route == "finalize"
    assert decision.reason == "submit_answer"
