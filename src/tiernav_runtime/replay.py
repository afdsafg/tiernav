"""Replay append-only event logs into materialized episode state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, StrictBool, TypeAdapter, ValidationError

from .contracts import EpisodeRequest, EpisodeState, NonNegativeInt, Observation
from .events import EpisodeEvent


_NON_NEGATIVE_INT_ADAPTER = TypeAdapter(NonNegativeInt)


class LegacyEpisodeStart(BaseModel):
    """Strict fallback schema for legacy start events without embedded requests."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    task_name: str
    task_mode: str
    prompt: str


class ToolResultReceivedPayload(BaseModel):
    """Strict payload schema for tool result events."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    step_index: Optional[NonNegativeInt] = None


class PolicyTransitionedPayload(BaseModel):
    """Strict payload schema for policy transition events."""

    model_config = ConfigDict(extra="forbid")

    failure_type: str


class EpisodeEndedPayload(BaseModel):
    """Strict payload schema for terminal episode events."""

    model_config = ConfigDict(extra="forbid")

    success: StrictBool
    answer: str = ""
    round_index: Optional[NonNegativeInt] = None
    step_index: Optional[NonNegativeInt] = None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key: {key}")
        obj[key] = value
    return obj


def _load_json_object(line: str) -> dict[str, Any]:
    value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError("event log line must be a JSON object")
    return value


def _load_events(path: str | Path) -> list[EpisodeEvent]:
    events: list[EpisodeEvent] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(EpisodeEvent.model_validate(_load_json_object(line)))
    sequences = [event.sequence for event in events]
    if any(current <= previous for previous, current in zip(sequences, sequences[1:])):
        raise ValueError(f"event sequence is out of order: {sequences}")
    return events


def _require_bool(payload: dict[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a bool")


def _optional_str(payload: dict[str, Any], field_name: str, current: str) -> str:
    if field_name not in payload:
        return current
    return _require_str(payload, field_name)


def _require_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a str")


def _require_episode_match(event: EpisodeEvent, episode_id: str) -> None:
    if event.episode_id != episode_id:
        raise ValueError(
            f"event episode_id {event.episode_id!r} does not match expected episode_id {episode_id!r}"
        )


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


def _require_not_terminal(state: EpisodeState, event_type: str) -> None:
    if state.terminal:
        raise ValueError(f"cannot apply {event_type} after terminal state")


def replay_events(path: str | Path) -> EpisodeState:
    """Rebuild materialized state from an event log."""

    events = _load_events(path)
    if not events:
        raise ValueError("cannot replay an empty event log")

    state: EpisodeState | None = None
    expected_episode_id: str | None = None
    for event in events:
        if event.event_type == "episode_started":
            if state is not None:
                raise ValueError("event log contains repeated episode_started")
            if "request" in event.payload:
                request_payload = event.payload["request"]
                request = EpisodeRequest.model_validate(request_payload)
                expected_episode_id = request.episode_id
                _require_episode_match(event, expected_episode_id)
                state = EpisodeState(
                    episode_id=request.episode_id,
                    scene_id=request.scene_id,
                    task_name=request.task_name,
                    task_mode=request.task_mode,
                    prompt=request.prompt,
                    pose=request.initial_pose,
                )
            else:
                legacy_start = LegacyEpisodeStart.model_validate(event.payload)
                expected_episode_id = event.episode_id
                state = EpisodeState(
                    episode_id=event.episode_id,
                    scene_id=legacy_start.scene_id,
                    task_name=legacy_start.task_name,
                    task_mode=legacy_start.task_mode,
                    prompt=legacy_start.prompt,
                )
        elif state is None:
            raise ValueError(f"event log starts with {event.event_type}, not episode_started")
        else:
            _require_episode_match(event, expected_episode_id or state.episode_id)
            _require_not_terminal(state, event.event_type)
            if event.event_type == "tool_result_received":
                payload = ToolResultReceivedPayload.model_validate(event.payload)
                state.last_observation = payload.observation
                if payload.step_index is not None:
                    state.step_index = payload.step_index
            elif event.event_type == "policy_transitioned":
                payload = PolicyTransitionedPayload.model_validate(event.payload)
                state.failure_type = payload.failure_type
            elif event.event_type == "episode_ended":
                payload = EpisodeEndedPayload.model_validate(event.payload)
                state.terminal = True
                state.success = payload.success
                state.answer = payload.answer
                if payload.round_index is not None:
                    state.round_index = payload.round_index
                if payload.step_index is not None:
                    state.step_index = payload.step_index
            else:
                raise ValueError(f"unsupported event_type: {event.event_type}")

    if state is None:
        raise ValueError("event log did not contain episode_started")
    return state
