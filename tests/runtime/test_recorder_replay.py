"""Tests for append-only runtime events and replay."""
import json

from src.tiernav_runtime.contracts import EpisodeRequest, EpisodeState, Observation
from src.tiernav_runtime.events import EpisodeEvent, make_event
from src.tiernav_runtime.recorder import EpisodeRecorder
from src.tiernav_runtime.replay import replay_events


def _request() -> EpisodeRequest:
    return EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is on the table?",
        output_dir="/tmp/tiernav",
    )


def test_make_event_has_schema_version_and_sequence():
    event = make_event(
        episode_id="ep-1",
        event_type="episode_started",
        sequence=1,
        payload={"scene_id": "scene"},
    )

    assert event.schema_version == "tiernav.runtime.v1"
    assert event.sequence == 1
    assert event.event_type == "episode_started"


def test_recorder_writes_jsonl_append_only(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = EpisodeRecorder(path)

    recorder.append(make_event("ep-1", "episode_started", 1, {"scene_id": "scene"}))
    recorder.append(make_event("ep-1", "episode_ended", 2, {"success": True}))

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "episode_started"
    assert json.loads(lines[1])["event_type"] == "episode_ended"


def test_replay_reconstructs_materialized_state(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = EpisodeRecorder(path)
    req = _request()

    recorder.append(make_event(req.episode_id, "episode_started", 1, {"request": req.model_dump(mode="json")}))
    recorder.append(make_event(req.episode_id, "tool_result_received", 2, {
        "observation": Observation(summary="Saw a mug.", image_ids=["snap-1"]).model_dump(mode="json"),
        "step_index": 1,
    }))
    recorder.append(make_event(req.episode_id, "episode_ended", 3, {
        "success": True,
        "answer": "mug",
        "round_index": 2,
        "step_index": 1,
    }))

    state = replay_events(path)

    assert isinstance(state, EpisodeState)
    assert state.episode_id == "ep-1"
    assert state.last_observation.summary == "Saw a mug."
    assert state.success is True
    assert state.answer == "mug"
    assert state.round_index == 2


def test_replay_rejects_out_of_order_sequences(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join([
            make_event("ep-1", "episode_started", 2, {}).model_dump_json(),
            make_event("ep-1", "episode_ended", 1, {}).model_dump_json(),
        ]) + "\n",
        encoding="utf-8",
    )

    try:
        replay_events(path)
    except ValueError as exc:
        assert "sequence" in str(exc)
    else:
        raise AssertionError("replay accepted out-of-order events")
