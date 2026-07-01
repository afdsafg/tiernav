"""Append-only JSONL recorder for runtime events."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import ContextSection
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


class PromptAuditRecorder:
    """Write per-round prompt sections (with full content) as JSONL.

    Each line is one compile round, containing the episode's sections with
    full content for post-hoc prompt auditing (token analysis, cache-break
    inspection, debugging).
    """

    def __init__(self, output_dir: str | Path):
        self.dir = Path(output_dir) / "prompt_audit"
        self.dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        episode_id: str,
        round_index: int,
        step_index: int,
        sections: list[ContextSection],
    ) -> None:
        path = self.dir / f"{episode_id}.jsonl"
        entry = {
            "round": round_index,
            "step": step_index,
            "sections": [
                {
                    "name": s.name,
                    "content": s.content,
                    "hash": s.content_hash,
                    "tokens": s.token_estimate,
                    "cacheable": s.cacheable,
                    "cache_break": s.cache_break,
                }
                for s in sections
            ],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
