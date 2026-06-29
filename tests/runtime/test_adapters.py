"""Tests for AEQA and GOATBench task adapters."""
from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from src.tiernav_runtime.adapters import AEQATaskAdapter, GOATBenchTaskAdapter
from src.tiernav_runtime.contracts import EpisodeRequest, EpisodeResult, TaskMode


# ---------------------------------------------------------------------------
# AEQA
# ---------------------------------------------------------------------------


def test_aeqa_builds_question_answering_request():
    adapter = AEQATaskAdapter()

    request = adapter.to_request(
        scene_id="scene-001",
        question_id="q-42",
        question="What color is the chair?",
        output_dir="/tmp/tiernav/out",
    )

    assert isinstance(request, EpisodeRequest)
    assert request.episode_id == "q-42"
    assert request.scene_id == "scene-001"
    assert request.task_name == "aeqa"
    assert request.task_mode == TaskMode.QUESTION_ANSWERING
    assert request.prompt == "What color is the chair?"
    assert request.output_dir == "/tmp/tiernav/out"
    assert request.initial_pose == {}


def test_aeqa_task_name_attribute():
    assert AEQATaskAdapter.task_name == "aeqa"


def test_aeqa_preserves_initial_pose():
    adapter = AEQATaskAdapter()
    pose = {"x": 1.5, "y": -0.5, "z": 0.0, "yaw": 0.25}

    request = adapter.to_request(
        scene_id="scene-001",
        question_id="q-1",
        question="Where is the lamp?",
        output_dir="/tmp/out",
        initial_pose=pose,
    )

    assert request.initial_pose == pose


def test_aeqa_exports_logger_payload():
    adapter = AEQATaskAdapter()
    result = EpisodeResult(
        episode_id="q-42",
        scene_id="scene-001",
        task_name="aeqa",
        task_mode="question_answering",
        success=True,
        answer="red",
        steps_taken=4,
        rounds_used=2,
        path_length=3.5,
        failure_type="",
        error="",
    )

    payload = adapter.to_eval_payload(result)

    assert payload["question_id"] == "q-42"
    assert payload["scene_id"] == "scene-001"
    assert payload["answer"] == "red"
    assert payload["success"] is True
    assert payload["steps_taken"] == 4
    assert payload["rounds_used"] == 2
    assert payload["path_length"] == 3.5
    assert payload["error"] == ""


def test_aeqa_payload_is_plain_dict_not_pydantic_model():
    adapter = AEQATaskAdapter()
    result = EpisodeResult(
        episode_id="q-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        success=False,
        error="timeout",
    )

    payload = adapter.to_eval_payload(result)

    assert type(payload) is dict
    assert not isinstance(payload, BaseModel)


def test_aeqa_adapter_builds_episode_request_without_cross_episode_memory():
    """Two EpisodeRequests from the same AEQA adapter are independent.

    Each question gets its own episode_id, prompt, and output_dir — no shared
    state carries over between q1 and q2.
    """
    adapter = AEQATaskAdapter()

    req1 = adapter.to_request(
        scene_id="scene-001",
        question_id="q-1",
        question="What color is the chair?",
        output_dir="/tmp/out/q1",
    )
    req2 = adapter.to_request(
        scene_id="scene-001",
        question_id="q-2",
        question="Where is the table?",
        output_dir="/tmp/out/q2",
    )

    assert req1.episode_id == "q-1"
    assert req2.episode_id == "q-2"
    assert req1.prompt == "What color is the chair?"
    assert req2.prompt == "Where is the table?"
    assert req1.task_mode == TaskMode.QUESTION_ANSWERING
    assert req2.task_mode == TaskMode.QUESTION_ANSWERING
    assert req1.output_dir == "/tmp/out/q1"
    assert req2.output_dir == "/tmp/out/q2"


def test_aeqa_to_eval_payload_preserves_fields():
    """AEQA to_eval_payload exports expected keys for downstream eval scripts."""
    adapter = AEQATaskAdapter()
    result = EpisodeResult(
        episode_id="q-7",
        scene_id="scene-3",
        task_name="aeqa",
        task_mode="question_answering",
        success=True,
        answer="blue",
        steps_taken=5,
        path_length=12.3,
        error="",
        rounds_used=3,
    )

    payload = adapter.to_eval_payload(result)

    expected_keys = {
        "question_id",
        "scene_id",
        "answer",
        "success",
        "steps_taken",
        "path_length",
        "error",
    }
    for key in expected_keys:
        assert key in payload, f"missing key: {key}"
    assert payload["question_id"] == "q-7"
    assert payload["scene_id"] == "scene-3"
    assert payload["answer"] == "blue"
    assert payload["success"] is True
    assert payload["steps_taken"] == 5
    assert payload["path_length"] == 12.3


# ---------------------------------------------------------------------------
# GOATBench
# ---------------------------------------------------------------------------


def test_goatbench_task_name_attribute():
    assert GOATBenchTaskAdapter.task_name == "goatbench"


def test_goatbench_builds_navigation_request_without_truth_leak():
    adapter = GOATBenchTaskAdapter()

    request = adapter.to_request(
        scene_id="scene-9",
        episode_id="ep-001",
        subtask_index=2,
        goal_type="object",
        goal_description="the red chair",
        output_dir="/tmp/tiernav/goat",
    )

    assert isinstance(request, EpisodeRequest)
    assert request.episode_id == "ep-001_2"
    assert request.scene_id == "scene-9"
    assert request.task_name == "goatbench"
    assert request.task_mode == TaskMode.GOAL_NAVIGATION
    assert request.prompt == "Navigate to the red chair"
    assert request.output_dir == "/tmp/tiernav/goat"
    assert request.initial_pose == {}

    meta = request.goal_metadata
    assert meta["goal_type"] == "object"
    assert meta["goal_description"] == "the red chair"
    assert meta["subtask_index"] == 2


def test_goatbench_goal_metadata_has_no_truth_leak_fields():
    adapter = GOATBenchTaskAdapter()

    request = adapter.to_request(
        scene_id="scene-9",
        episode_id="ep-001",
        subtask_index=0,
        goal_type="room",
        goal_description="kitchen",
        output_dir="/tmp/out",
    )

    meta = request.goal_metadata
    forbidden = {"truth", "answer", "success", "path", "ground_truth", "gt_path", "target_path"}
    leak_keys = forbidden & set(meta.keys())
    assert not leak_keys, f"goal_metadata leaks truth fields: {leak_keys}"


def test_goatbench_preserves_initial_pose():
    adapter = GOATBenchTaskAdapter()
    pose = {"x": 0.0, "y": 0.0, "z": 1.0}

    request = adapter.to_request(
        scene_id="scene-9",
        episode_id="ep-001",
        subtask_index=1,
        goal_type="object",
        goal_description="lamp",
        output_dir="/tmp/out",
        initial_pose=pose,
    )

    assert request.initial_pose == pose


def test_goatbench_exports_navigation_payload():
    adapter = GOATBenchTaskAdapter()
    result = EpisodeResult(
        episode_id="ep-001_2",
        scene_id="scene-9",
        task_name="goatbench",
        task_mode="goal_navigation",
        success=False,
        answer="",
        steps_taken=12,
        rounds_used=5,
        path_length=7.25,
        failure_type="timeout",
        error="max_rounds_exceeded",
    )

    payload = adapter.to_eval_payload(result)

    assert payload["subtask_id"] == "ep-001_2"
    assert payload["scene_id"] == "scene-9"
    assert payload["success"] is False
    assert payload["answer"] == ""
    assert payload["steps_taken"] == 12
    assert payload["rounds_used"] == 5
    assert payload["path_length"] == 7.25
    assert payload["failure_type"] == "timeout"
    assert payload["error"] == "max_rounds_exceeded"


def test_goatbench_adapter_threads_subtask_context_and_goal_metadata():
    """GOATBench adapter threads subtasks inside one episode with shared episode_id.

    start_episode opens a long-lived session; run_subtask produces an
    EpisodeRequest whose episode_id is the per-episode id (not a per-subtask
    composite), so memory services see the same session across subtasks.
    goal_metadata carries goal_description, goal_type, and subtask_index
    for the planner without leaking success/truth fields.
    """
    adapter = GOATBenchTaskAdapter()
    adapter.start_episode("ep-42", scene_id="scene-9", output_dir="/tmp/goat")

    req0 = adapter.run_subtask(
        subtask_index=0,
        goal_type="object",
        goal_description="the red chair",
    )
    req1 = adapter.run_subtask(
        subtask_index=1,
        goal_type="object",
        goal_description="the round table",
        initial_pose={"x": 1.0, "y": 2.0, "z": 0.0},
    )

    # Both subtasks share the episode-level id — no per-subtask composite.
    assert req0.episode_id == "ep-42"
    assert req1.episode_id == "ep-42"
    assert req0.scene_id == "scene-9"
    assert req1.scene_id == "scene-9"
    assert req0.task_mode == TaskMode.GOAL_NAVIGATION
    assert req1.task_mode == TaskMode.GOAL_NAVIGATION

    # goal_metadata carries per-subtask identity without truth leakage.
    m0 = req0.goal_metadata
    assert m0["goal_type"] == "object"
    assert m0["goal_description"] == "the red chair"
    assert m0["subtask_index"] == 0

    m1 = req1.goal_metadata
    assert m1["goal_type"] == "object"
    assert m1["goal_description"] == "the round table"
    assert m1["subtask_index"] == 1

    assert req1.initial_pose == {"x": 1.0, "y": 2.0, "z": 0.0}


def test_goatbench_to_eval_payload_includes_new_fields():
    """GOATBench to_eval_payload forwards distance_to_goal and submit_was_explicit."""
    adapter = GOATBenchTaskAdapter()
    result = EpisodeResult(
        episode_id="ep-42",
        scene_id="scene-9",
        task_name="goatbench",
        task_mode="goal_navigation",
        success=False,
        failure_type="distance_exceeded",
        distance_to_goal=3.5,
        submit_was_explicit=True,
    )

    payload = adapter.to_eval_payload(result)

    assert payload["distance_to_goal"] == 3.5
    assert payload["submit_was_explicit"] is True


def test_goatbench_payload_is_plain_dict_not_pydantic_model():
    adapter = GOATBenchTaskAdapter()
    result = EpisodeResult(
        episode_id="ep-001_0",
        scene_id="scene-9",
        task_name="goatbench",
        task_mode="goal_navigation",
        success=True,
    )

    payload = adapter.to_eval_payload(result)

    assert type(payload) is dict
    assert not isinstance(payload, BaseModel)


# ---------------------------------------------------------------------------
# No external-service / no-langgraph guard
# ---------------------------------------------------------------------------


def test_adapters_module_has_no_external_dependencies():
    import src.tiernav_runtime.adapters as adapters_mod

    src = inspect.getsource(adapters_mod)
    forbidden = ["langgraph", "requests", "httpx", "openai", "anthropic", "aiohttp"]
    for token in forbidden:
        assert token not in src, f"adapters.py references forbidden dependency: {token}"
