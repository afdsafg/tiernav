"""Replay append-only event logs into materialized episode state."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import EpisodeRequest, EpisodeState, Observation
from .events import EpisodeEvent


def _load_events(path: str | Path) -> list[EpisodeEvent]:
    events: list[EpisodeEvent] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(EpisodeEvent.model_validate(json.loads(line)))
    sequences = [event.sequence for event in events]
    if sequences != sorted(sequences):
        raise ValueError(f"event sequence is out of order: {sequences}")
    return events


def replay_events(path: str | Path) -> EpisodeState:
    """Rebuild materialized state from an event log."""

    events = _load_events(path)
    if not events:
        raise ValueError("cannot replay an empty event log")

    state: EpisodeState | None = None
    for event in events:
        if event.event_type == "episode_started":
            request_payload = event.payload.get("request")
            if request_payload:
                request = EpisodeRequest.model_validate(request_payload)
                state = EpisodeState(
                    episode_id=request.episode_id,
                    scene_id=request.scene_id,
                    task_name=request.task_name,
                    task_mode=request.task_mode,
                    prompt=request.prompt,
                    pose=request.initial_pose,
                )
            else:
                state = EpisodeState(
                    episode_id=event.episode_id,
                    scene_id=str(event.payload.get("scene_id", "")),
                    task_name=str(event.payload.get("task_name", "")),
                    task_mode=event.payload.get("task_mode", "question_answering"),
                    prompt=str(event.payload.get("prompt", "")),
                )
        elif state is None:
            raise ValueError(f"event log starts with {event.event_type}, not episode_started")
        elif event.event_type == "tool_result_received":
            if "observation" in event.payload:
                state.last_observation = Observation.model_validate(event.payload["observation"])
            state.step_index = int(event.payload.get("step_index", state.step_index))
        elif event.event_type == "policy_transitioned":
            state.failure_type = str(event.payload.get("failure_type", state.failure_type))
        elif event.event_type == "episode_ended":
            state.terminal = True
            state.success = bool(event.payload.get("success", False))
            state.answer = str(event.payload.get("answer", ""))
            state.round_index = int(event.payload.get("round_index", state.round_index))
            state.step_index = int(event.payload.get("step_index", state.step_index))

    if state is None:
        raise ValueError("event log did not contain episode_started")
    return state
