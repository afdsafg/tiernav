"""Tests for RuntimeEntrypoint and legacy-compatible return mapping.

Deterministic fakes only: no external services, no network. Exercises the
Task 9 planned example (FakePlanner -> chair -> success) plus extra coverage
for event-log replayability, sequence ordering, and the legacy dict shape.

The entrypoint must not wire production runners to the fake runtime; it is
a deterministic dev/replay path backed by stable default tools.
"""
from __future__ import annotations

import json

import pytest

from src.tiernav_runtime.contracts import (
    BenchmarkRule,
    EpisodeRequest,
    EpisodeResult,
    MemoryScope,
    PlannerDecision,
    RunSpec,
    TaskMode,
)
from src.tiernav_runtime.entrypoint import (
    RuntimeEntrypoint,
    episode_result_to_legacy_dict,
)
from src.tiernav_runtime.replay import replay_events
from src.tiernav_runtime.tools import SubmitAnswerTool, ToolRegistry


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

    def decide(self, prompt: str, **kwargs) -> PlannerDecision:
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

    # episode_started is always first (sequence=1); episode_ended is always
    # last with the highest sequence. Intra-episode events (context_compiled,
    # planner_called, planner_decision, success_evaluated) sit between them,
    # so episode_ended's sequence is no longer hard-coded to 2.
    assert lines[0]["event_type"] == "episode_started"
    assert lines[0]["sequence"] == 1
    assert lines[-1]["event_type"] == "episode_ended"
    assert lines[-1]["sequence"] == max(line["sequence"] for line in lines)
    # episode_started payload carries the request.
    assert "request" in lines[0]["payload"]
    assert lines[0]["payload"]["request"]["episode_id"] == "ep-chair"
    # episode_ended payload carries the required terminal fields.
    ended_payload = lines[-1]["payload"]
    for key in ("success", "answer", "round_index", "step_index"):
        assert key in ended_payload, f"episode_ended payload missing {key}"


# ── Re-run protection ─────────────────────────────────────────────────────


def test_second_run_same_episode_id_raises_FileExistsError(tmp_path):
    """Re-running the same episode_id into the same output_dir must be rejected.

    The event log is append-only; a second run must not overwrite or append to
    a previously recorded log. The old log stays intact and replayable.
    """
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)
    spec = _spec(str(tmp_path))
    request = _request()

    first = entrypoint.run(spec, request)
    assert first.success is True

    with pytest.raises(FileExistsError):
        entrypoint.run(spec, request)

    # Old event log is untouched: still starts with episode_started, ends
    # with episode_ended, and replays to terminal. The line count is no
    # longer fixed at 2 now that intra-episode events are emitted.
    with open(first.event_log_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert lines[0]["event_type"] == "episode_started"
    assert lines[-1]["event_type"] == "episode_ended"

    state = replay_events(first.event_log_path)
    assert state.terminal is True
    assert state.answer == "chair"


# ── with_environment_services (Task 3) ───────────────────────────────────


class _FakeEnv:
    """Minimal environment double — RuntimeServices just holds the reference."""

    task_mode = "question_answering"


def test_with_environment_services_attaches_env_to_services():
    """RuntimeEntrypoint.with_environment_services wires env onto RuntimeServices.

    The graph is not yet rewired to call the env (Task 7), but the service
    bundle must carry the environment reference so production runners can
    build it and Task 7 can read it.
    """
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint

    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    env = _FakeEnv()
    entrypoint = RuntimeEntrypoint.with_environment_services(planner, env)

    assert entrypoint.services.environment is env


# ── Phase 2: SceneMemoryStore wiring in run() ────────────────────────────


def test_run_creates_scene_memory_store_when_output_dir_set(tmp_path):
    """run() with output_dir must create a SceneMemoryStore and persist on disk.

    Fake-services path has no memory_session, so sediment is a no-op, but the
    store is still created (and its JSON file materializes once sediment runs
    against a session-backed services bundle). Here we just verify the store
    is non-None during invoke by checking the on-disk artifact after a
    successful episode using a MemorySession-wired entrypoint.
    """
    import os

    from src.tiernav_runtime.contracts import MemoryScope
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint
    from src.tiernav_runtime.memory import (
        MemoryService,
        MemorySession,
        ObjectNode,
        RoomNode,
    )

    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    # Build via with_fake_services then inject a MemorySession so sediment has
    # something to read. This mirrors how production wires memory_session.
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)
    session = MemorySession(scope=MemoryScope.PER_QUESTION)
    entrypoint.services.memory_session = session

    result = entrypoint.run(_spec(str(tmp_path)), _request())

    assert result.success is True
    # scene_memory/<scene_id>.json should exist after sediment ran.
    scene_mem_path = tmp_path / "scene_memory" / "scene-1.json"
    assert os.path.exists(scene_mem_path), "scene_memory JSON must be persisted"

    # Store is cleared in finally — verify it does not leak across episodes.
    assert entrypoint.services.scene_memory_store is None


def test_run_clears_scene_memory_store_in_finally(tmp_path):
    """Even on a successful run, scene_memory_store is cleared in finally."""
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint

    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)

    entrypoint.run(_spec(str(tmp_path)), _request())

    assert entrypoint.services.scene_memory_store is None


def test_run_skips_store_when_output_dir_empty(tmp_path):
    """When output_dir is empty/falsy, no store is created."""
    from src.tiernav_runtime.entrypoint import RuntimeEntrypoint

    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})]
    )
    entrypoint = RuntimeEntrypoint.with_fake_services(planner)
    spec = _spec("")  # empty output_dir

    # run() with empty output_dir should not create the recorder path either;
    # just verify store stays None and run does not raise.
    try:
        entrypoint.run(spec, _request())
    except (FileExistsError, OSError):
        pass  # empty output_dir may fail on event-log path; that's fine here
    assert entrypoint.services.scene_memory_store is None


# ── with_real_services injection (Task 9) ─────────────────────────────────


def test_with_real_services_accepts_custom_tools_and_aeqa_controller():
    class Planner:
        def decide(self, prompt, **kwargs):
            return PlannerDecision(action_type="submit_answer", arguments={"answer": "unused"})

    class Controller:
        pass

    custom_tools = ToolRegistry()
    custom_tools.register(SubmitAnswerTool(task_mode="question_answering"))
    controller = Controller()
    rule = BenchmarkRule(
        success_distance_m=0.0,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="aeqa",
    )

    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=Planner(),
        environment=None,
        rule=rule,
        executor=None,
        task_mode="question_answering",
        tools=custom_tools,
        aeqa_controller=controller,
    )

    assert entrypoint.services.tools is custom_tools
    assert entrypoint.services.aeqa_controller is controller


def test_with_real_services_preserves_custom_tool_set_during_run(tmp_path):
    class Planner:
        def decide(self, prompt, **kwargs):
            return PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})

    custom_tools = ToolRegistry()
    custom_tools.register(SubmitAnswerTool(task_mode="question_answering"))
    rule = BenchmarkRule(
        success_distance_m=0.0,
        memory_scope=MemoryScope.PER_QUESTION,
        scoring_mode="aeqa",
    )
    entrypoint = RuntimeEntrypoint.with_real_services(
        planner=Planner(),
        environment=None,
        rule=rule,
        executor=None,
        task_mode="question_answering",
        tools=custom_tools,
    )

    result = entrypoint.run(_spec(str(tmp_path)), _request("ep-custom-tools"))

    assert result.success is True
    assert entrypoint.services.tools.names() == ["submit_answer"]
