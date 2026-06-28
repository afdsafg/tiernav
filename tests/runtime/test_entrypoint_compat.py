"""Tests for RuntimeEntrypoint and legacy-compatible return mapping.

Deterministic fakes only: no external services, no network. Exercises the
Task 9 planned example (FakePlanner -> chair -> success) plus extra coverage
for event-log replayability, sequence ordering, and the legacy dict shape.

The entrypoint must not wire production runners to the fake runtime; it is
a deterministic dev/replay path backed by stable default tools.
"""
from __future__ import annotations

import json

from src.tiernav_runtime.contracts import (
    EpisodeRequest,
    EpisodeResult,
    PlannerDecision,
    RunSpec,
    TaskMode,
)
from src.tiernav_runtime.entrypoint import (
    RuntimeEntrypoint,
    episode_result_to_legacy_dict,
)
from src.tiernav_runtime.replay import replay_events


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakePlanner:
    """Deterministic planner that replays a scripted decision sequence.

    Returns the last decision once the script is exhausted so the policy can
    route to a terminal node.
    """

    def __init__(self, script: list[PlannerDecision]) -> None:
        if not script:
            raise ValueError("script must be non-empty")
        self._script = list(script)
        self._i = 0

    def decide(self, prompt: str) -> PlannerDecision:
        decision = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return decision


def _spec(output_dir: str) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        task_name="aeqa",
        dataset_split="dev",
        output_dir=output_dir,
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
    )


def _request(episode_id: str = "ep-chair") -> EpisodeRequest:
    return EpisodeRequest(
        episode_id=episode_id,
        scene_id="scene-1",
        task_name="aeqa",
        task_mode=TaskMode.QUESTION_ANSWERING,
        prompt="What is sitting in the chair?",
    )


# ── Planned example ───────────────────────────────────────────────────────


def test_with_fake_services_returns_success_with_answer(tmp_path):
    """RuntimeEntrypoint.with_fake_services(FakePlanner).run -> success, chair, non-empty log."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)
    result = entrypoint.run(_spec(str(tmp_path)), _request())

    assert isinstance(result, EpisodeResult)
    assert result.success is True
    assert result.answer == "chair"
    assert result.event_log_path, "event_log_path must be non-empty"


# ── Legacy dict mapping ──────────────────────────────────────────────────


def test_episode_result_to_legacy_dict_preserves_fields():
    result = EpisodeResult(
        episode_id="q-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode=TaskMode.QUESTION_ANSWERING,
        success=True,
        answer="chair",
        steps_taken=3,
        rounds_used=2,
        path_length=1.5,
        failure_type="",
        error="",
        event_log_path="/tmp/ep/events.jsonl",
    )

    legacy = episode_result_to_legacy_dict(result, question="Where is the chair?")

    assert legacy["scene_id"] == "scene-1"
    assert legacy["question_id"] == "q-1"
    assert legacy["question"] == "Where is the chair?"
    assert legacy["answer"] == "chair"
    assert legacy["success"] is True
    assert legacy["steps_taken"] == 3
    assert legacy["rounds_used"] == 2
    assert legacy["path_length"] == 1.5
    assert legacy["error"] == ""
    assert legacy["event_log_path"] == "/tmp/ep/events.jsonl"
    assert legacy["failure_type"] == ""


def test_legacy_dict_defaults_n_snapshots_to_zero():
    result = EpisodeResult(
        episode_id="q-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode=TaskMode.QUESTION_ANSWERING,
        success=False,
    )
    legacy = episode_result_to_legacy_dict(result)

    assert legacy["n_filtered_snapshots"] == 0
    assert legacy["n_total_snapshots"] == 0


# ── Event log ────────────────────────────────────────────────────────────


def test_event_log_exists_and_replays_to_terminal_state(tmp_path):
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)
    result = entrypoint.run(_spec(str(tmp_path)), _request())

    import os

    assert os.path.exists(result.event_log_path)

    state = replay_events(result.event_log_path)
    assert state.terminal is True
    assert state.answer == "chair"


def test_event_log_has_started_and_ended_with_sequences_1_and_2(tmp_path):
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)
    result = entrypoint.run(_spec(str(tmp_path)), _request())

    with open(result.event_log_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]

    assert len(lines) == 2
    assert lines[0]["event_type"] == "episode_started"
    assert lines[0]["sequence"] == 1
    assert lines[1]["event_type"] == "episode_ended"
    assert lines[1]["sequence"] == 2
    # episode_started payload carries the request.
    assert "request" in lines[0]["payload"]
    assert lines[0]["payload"]["request"]["episode_id"] == "ep-chair"
    # episode_ended payload carries the required terminal fields.
    ended_payload = lines[1]["payload"]
    for key in ("success", "answer", "round_index", "step_index"):
        assert key in ended_payload, f"episode_ended payload missing {key}"
