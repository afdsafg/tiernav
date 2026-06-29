"""Tests for benchmark-specific success evaluation.

AEQA: runtime success = answer submitted (quality is external LLM-Match).
GOATBench: success requires explicit terminal submit AND distance-to-goal
within ``rule.success_distance_m``. Snapshot presence alone never marks
GOATBench success.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tiernav_runtime.contracts import (
    BenchmarkRule,
    EpisodeResult,
    MemoryScope,
    TaskMode,
)
from src.tiernav_runtime.success import SuccessEvaluator, SuccessVerdict


# --- Benchmark rule fixtures ------------------------------------------------


def _aeqa_rule() -> BenchmarkRule:
    return BenchmarkRule(
        success_distance_m=0.0,
        requires_explicit_stop=False,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="answer_match",
    )


def _goatbench_rule(success_distance_m: float = 1.0) -> BenchmarkRule:
    return BenchmarkRule(
        success_distance_m=success_distance_m,
        requires_explicit_stop=True,
        memory_scope=MemoryScope.SUBTASK_SEQUENCE,
        scoring_mode="distance",
    )


# --- AEQA -------------------------------------------------------------------


def test_aeqa_success_requires_answer_submission():
    evaluator = SuccessEvaluator(_aeqa_rule())

    verdict = evaluator.evaluate(
        TaskMode.QUESTION_ANSWERING,
        submitted_explicitly=True,
        answer="chair",
    )
    assert verdict.success is True
    assert verdict.reason == "answer_submitted"


def test_aeqa_no_explicit_submit_fails():
    evaluator = SuccessEvaluator(_aeqa_rule())

    verdict = evaluator.evaluate(
        TaskMode.QUESTION_ANSWERING,
        submitted_explicitly=False,
        answer="chair",
    )
    assert verdict.success is False
    assert verdict.reason == "no_explicit_submit"


def test_aeqa_empty_answer_fails():
    evaluator = SuccessEvaluator(_aeqa_rule())

    verdict = evaluator.evaluate(
        TaskMode.QUESTION_ANSWERING,
        submitted_explicitly=True,
        answer="",
    )
    assert verdict.success is False
    assert verdict.reason == "no_answer"


def test_aeqa_distance_is_irrelevant():
    """Distance must not affect AEQA verdicts even if supplied."""
    evaluator = SuccessEvaluator(_aeqa_rule())

    verdict = evaluator.evaluate(
        TaskMode.QUESTION_ANSWERING,
        submitted_explicitly=True,
        answer="chair",
        distance_to_goal=99.0,
    )
    assert verdict.success is True
    assert verdict.reason == "answer_submitted"


# --- GOATBench --------------------------------------------------------------


def test_goatbench_success_requires_explicit_submit_and_distance():
    evaluator = SuccessEvaluator(_goatbench_rule(success_distance_m=1.0))

    # Within threshold + explicit submit -> success.
    verdict = evaluator.evaluate(
        TaskMode.GOAL_NAVIGATION,
        submitted_explicitly=True,
        distance_to_goal=0.5,
    )
    assert verdict.success is True
    assert verdict.reason == "distance_within_threshold"


def test_goatbench_distance_exceeded_fails():
    evaluator = SuccessEvaluator(_goatbench_rule(success_distance_m=1.0))

    verdict = evaluator.evaluate(
        TaskMode.GOAL_NAVIGATION,
        submitted_explicitly=True,
        distance_to_goal=1.5,
    )
    assert verdict.success is False
    assert verdict.reason == "distance_exceeded"


def test_goatbench_no_explicit_submit_fails_even_if_distance_good():
    evaluator = SuccessEvaluator(_goatbench_rule(success_distance_m=1.0))

    verdict = evaluator.evaluate(
        TaskMode.GOAL_NAVIGATION,
        submitted_explicitly=False,
        distance_to_goal=0.3,
    )
    assert verdict.success is False
    assert verdict.reason == "no_explicit_submit"


def test_goatbench_missing_distance_fails():
    evaluator = SuccessEvaluator(_goatbench_rule(success_distance_m=1.0))

    verdict = evaluator.evaluate(
        TaskMode.GOAL_NAVIGATION,
        submitted_explicitly=True,
        distance_to_goal=None,
    )
    assert verdict.success is False
    assert verdict.reason == "no_distance"


def test_goatbench_boundary_distance_is_success():
    """distance == success_distance_m is within threshold (<=)."""
    evaluator = SuccessEvaluator(_goatbench_rule(success_distance_m=1.0))

    verdict = evaluator.evaluate(
        TaskMode.GOAL_NAVIGATION,
        submitted_explicitly=True,
        distance_to_goal=1.0,
    )
    assert verdict.success is True
    assert verdict.reason == "distance_within_threshold"


# --- Snapshot presence does not imply success -------------------------------


def test_goatbench_evaluator_has_no_snapshot_argument():
    """The evaluator API must not accept a snapshot presence flag.

    GOATBench success cannot be inferred from snapshot presence alone; the
    evaluator signature intentionally omits any snapshot parameter. Calling
    evaluate() with a stray ``snapshot`` kwarg must raise TypeError.
    """
    evaluator = SuccessEvaluator(_goatbench_rule(success_distance_m=1.0))

    with pytest.raises(TypeError):
        evaluator.evaluate(  # type: ignore[call-arg]
            TaskMode.GOAL_NAVIGATION,
            submitted_explicitly=True,
            distance_to_goal=0.5,
            snapshot_present=True,
        )


# --- EpisodeResult distance fields ------------------------------------------


def test_episode_result_distance_fields_default_none_and_false():
    result = EpisodeResult(
        episode_id="ep-1",
        scene_id="scene",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        success=False,
    )
    assert result.distance_to_goal is None
    assert result.submit_was_explicit is False


def test_episode_result_distance_fields_settable():
    result = EpisodeResult(
        episode_id="ep-1",
        scene_id="scene",
        task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION,
        success=False,
        distance_to_goal=0.8,
        submit_was_explicit=True,
    )
    assert result.distance_to_goal == 0.8
    assert result.submit_was_explicit is True


def test_episode_result_rejects_negative_distance_to_goal():
    with pytest.raises(ValidationError):
        EpisodeResult(
            episode_id="ep-1",
            scene_id="scene",
            task_name="goatbench",
            task_mode=TaskMode.GOAL_NAVIGATION,
            success=False,
            distance_to_goal=-0.1,
        )


# --- Verdict model ----------------------------------------------------------


def test_success_verdict_carries_success_and_reason():
    verdict = SuccessVerdict(success=True, reason="answer_submitted")
    assert verdict.success is True
    assert verdict.reason == "answer_submitted"
