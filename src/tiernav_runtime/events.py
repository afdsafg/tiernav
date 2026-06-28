"""Append-only event envelopes for TierNav runtime."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import Field, field_validator

from .contracts import JsonObject, RuntimeModel, SCHEMA_VERSION, SchemaVersion


PositiveSequence = Annotated[int, Field(strict=True, gt=0)]


def _validate_utc_timestamp(value: Any) -> str:
    """Accept only UTC ISO-8601 / RFC3339 strings; reject numbers, naive, non-UTC offsets."""

    if not isinstance(value, str):
        raise ValueError("timestamp_utc must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp_utc must be a UTC ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp_utc must include a timezone")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp_utc must be UTC (offset +00:00 or Z)")
    return value


UTCTimestamp = Annotated[str, Field(strict=True)]


class EpisodeEvent(RuntimeModel):
    """One append-only event in an episode log."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    episode_id: str
    event_type: str
    sequence: PositiveSequence
    timestamp_utc: UTCTimestamp
    payload: JsonObject = Field(default_factory=dict)

    @field_validator("timestamp_utc")
    @classmethod
    def _timestamp_utc_must_be_utc_string(cls, value: Any) -> str:
        return _validate_utc_timestamp(value)


def make_event(
    episode_id: str,
    event_type: str,
    sequence: int,
    payload: JsonObject | None = None,
) -> EpisodeEvent:
    """Create a validated event envelope."""

    return EpisodeEvent(
        episode_id=episode_id,
        event_type=event_type,
        sequence=sequence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        payload={} if payload is None else payload,
    )
