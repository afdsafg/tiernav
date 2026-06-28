"""Tests for AEQA and GOATBench task adapters."""
from __future__ import annotations

import inspect
from typing import get_type_hints

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
    assert payload["failure_type"] == "timeout"
    assert payload["error"] == "max_rounds_exceeded"


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
