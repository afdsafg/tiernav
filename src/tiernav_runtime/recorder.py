"""Append-only JSONL recorder for runtime events."""
from __future__ import annotations

from pathlib import Path

from .events import EpisodeEvent


class EpisodeRecorder:
    """Write episode events as append-only JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: EpisodeEvent) -> None:
        """Append one event without rewriting existing lines."""

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")
