"""Append-only event envelopes for TierNav runtime."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from .contracts import RuntimeModel, SCHEMA_VERSION


class EpisodeEvent(RuntimeModel):
    """One append-only event in an episode log."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    event_type: str
    sequence: int
    timestamp_utc: str
    payload: dict[str, Any] = Field(default_factory=dict)


def make_event(
    episode_id: str,
    event_type: str,
    sequence: int,
    payload: dict[str, Any] | None = None,
) -> EpisodeEvent:
    """Create a validated event envelope."""

    return EpisodeEvent(
        episode_id=episode_id,
        event_type=event_type,
        sequence=sequence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        payload=payload or {},
    )
