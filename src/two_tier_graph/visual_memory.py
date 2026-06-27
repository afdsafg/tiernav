"""L0 visual memory index — always-in-prompt one-line-per-snapshot summary.

Borrowed from Claude Code's MEMORY.md ≤200-line index pattern. Closes the CLIP
embedding gap (agent_memory.py:50 computes but :62 bypasses): CLIP embedding
is accepted by `update` and reserved for future ordering/ranking use within the
index; the one-line-per-snapshot text itself stays lightweight and always-in-prompt.

Contract:
  - ≤1 line per snapshot, refreshed every `refresh_interval` rounds.
  - Dedup by `snapshot_id` — a snapshot already in the index is never re-added.
  - Failure-safe: any rebuild error keeps the last good `_text`; `get_index_text`
    always returns a str (empty string on corruption/initial state).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VisualMemoryIndex:
    """L0 index layer: ≤1 line per snapshot, always in prompt.

    `clip_embedding` is accepted by `update` (closing the agent_memory.py:50/62
    gap — embeddings were computed then discarded) and stored alongside the
    entry for future ordering/ranking without re-running CLIP. The always-in-prompt
    text itself is just the one-line summary, kept small.
    """

    refresh_interval: int = 3
    _entries: Dict[str, str] = field(default_factory=dict)  # snapshot_id -> index line
    _embeddings: Dict[str, Any] = field(default_factory=dict)  # snapshot_id -> clip_embedding
    _text: Optional[str] = None
    _last_rebuild_round: int = -1

    def should_rebuild(self, round_idx: int) -> bool:
        """True when round_idx is on the refresh cadence (default every 3)."""
        return round_idx % self.refresh_interval == 0

    def update(
        self,
        round_idx: int,
        pose: Tuple[float, ...],
        object_class: str,
        one_line_desc: str,
        snapshot_id: str,
        clip_embedding: Optional[Any] = None,
    ) -> None:
        """Add a snapshot entry. Dedup by snapshot_id — re-calling with the same
        id is a no-op (keeps the first line, preserves cross-round stability)."""
        if snapshot_id in self._entries:
            return  # dedup
        line = f"[R{round_idx}, pose={pose}, obj={object_class}] {one_line_desc}"
        self._entries[snapshot_id] = line
        if clip_embedding is not None:
            self._embeddings[snapshot_id] = clip_embedding
        self._rebuild_text()

    def _rebuild_text(self) -> None:
        """Rebuild the cached text. On failure keep the last good `_text`
        (fallback) — never raise."""
        try:
            self._text = "\n".join(self._entries.values())
        except Exception as e:
            logging.warning("L0 index rebuild failed: %s", e)
            # Keep last good _text as fallback

    def get_index_text(self) -> str:
        """Always returns a str — empty when empty/corrupt (graceful fallback)."""
        if self._text is None:
            return ""
        return self._text

    def to_state(self) -> dict:
        """Serialize for LangGraph state. `_embeddings` omitted — non-serializable
        numpy arrays; rebuilt from notebook snapshots on next rebuild."""
        return {
            "entries": dict(self._entries),
            "text": self._text,
            "last_rebuild_round": self._last_rebuild_round,
        }

    @classmethod
    def from_state(cls, state: dict, refresh_interval: int = 3) -> "VisualMemoryIndex":
        """Reconstruct from state dict. Missing keys → empty (first round)."""
        idx = cls(refresh_interval=refresh_interval)
        idx._entries = dict(state.get("entries", {}))
        idx._text = state.get("text", "")
        idx._last_rebuild_round = state.get("last_rebuild_round", -1)
        return idx


class CaptionStore:
    """L1 caption layer — disk-cached VLM captions per snapshot.

    Borrowed from Claude Code findRelevantMemories. Each snapshot gets a
    VLM caption (disk-cached at cache_dir/<snapshot_id>.txt).
    build_context_node retrieves top-K via CLIP similarity.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def put(self, snapshot_id: str, caption: str) -> None:
        """Write caption to disk cache."""
        path = os.path.join(self.cache_dir, f"{snapshot_id}.txt")
        with open(path, "w") as f:
            f.write(caption)

    def get(self, snapshot_id: str) -> Optional[str]:
        """Read caption from disk cache. Returns None if missing."""
        path = os.path.join(self.cache_dir, f"{snapshot_id}.txt")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return f.read().strip()

    def has(self, snapshot_id: str) -> bool:
        """True if a caption is cached for snapshot_id."""
        return os.path.exists(os.path.join(self.cache_dir, f"{snapshot_id}.txt"))

    def top_k(self, query: str, k: int = 3) -> List[str]:
        """Return top-K captions by CLIP similarity to query.

        TODO: wire real CLIP retrieval. For now returns all cached captions
        up to k (unranked).
        """
        captions: List[str] = []
        for fname in sorted(os.listdir(self.cache_dir)):
            if fname.endswith(".txt"):
                with open(os.path.join(self.cache_dir, fname)) as f:
                    captions.append(f.read().strip())
        return captions[:k]
