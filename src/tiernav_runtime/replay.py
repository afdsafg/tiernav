"""Replay append-only event logs into materialized episode state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .contracts import EpisodeRequest, EpisodeState, NonNegativeInt, Observation
from .events import EpisodeEvent


_NON_NEGATIVE_INT_ADAPTER = TypeAdapter(NonNegativeInt)


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


def _require_bool(payload: dict[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a bool")


def _require_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a str")


def _require_non_negative_int(payload: dict[str, Any], field_name: str) -> int:
    if field_name not in payload:
        raise ValueError(f"{field_name} is required")
    value = payload[field_name]
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative int")
    try:
        return _NON_NEGATIVE_INT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"{field_name} must be a non-negative int") from exc


def _optional_non_negative_int(payload: dict[str, Any], field_name: str, current: int) -> int:
    if field_name not in payload:
        return current
    return _require_non_negative_int(payload, field_name)


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
            state.step_index = _optional_non_negative_int(event.payload, "step_index", state.step_index)
        elif event.event_type == "policy_transitioned":
            state.failure_type = str(event.payload.get("failure_type", state.failure_type))
        elif event.event_type == "episode_ended":
            state.terminal = True
            state.success = _require_bool(event.payload, "success")
            state.answer = _require_str(event.payload, "answer") if "answer" in event.payload else ""
            state.round_index = _optional_non_negative_int(event.payload, "round_index", state.round_index)
            state.step_index = _optional_non_negative_int(event.payload, "step_index", state.step_index)

    if state is None:
        raise ValueError("event log did not contain episode_started")
    return state
