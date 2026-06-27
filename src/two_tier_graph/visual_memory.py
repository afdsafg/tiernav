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


class ImageRecallStore:
    """L2 image recall layer — token-budgeted original snapshot recall.

    Borrowed from Claude Code loadedNestedMemoryPaths LRU. Recalls original
    snapshots into prompt when planner requests visual verification.
    Hard token budget (default 3000 vision tokens, ~3 images).
    """

    def __init__(self, cache_dir: str, token_budget: int = 3000, tokens_per_image: int = 1000):
        self.cache_dir = cache_dir
        self.token_budget = token_budget
        self.tokens_per_image = tokens_per_image
        self._registry: Dict[str, dict] = {}  # snapshot_id -> {image_b64, tokens}

    def register(self, snapshot_id: str, image_b64: str, tokens: Optional[int] = None) -> None:
        """Register a snapshot as available for recall."""
        self._registry[snapshot_id] = {
            "image_b64": image_b64,
            "tokens": tokens if tokens is not None else self.tokens_per_image,
        }

    def select_for_recall(self, query: str, loaded_ids: set, k: Optional[int] = None) -> List[dict]:
        """Select snapshots for recall, respecting token budget + LRU dedup.

        Args:
            query: query text (TODO: CLIP ranking)
            loaded_ids: already-loaded snapshot IDs (skip these)
            k: max images (default: token_budget // tokens_per_image)

        Returns:
            list of {"snapshot_id": str, "image_b64": str, "tokens": int}
        """
        if k is None:
            k = self.token_budget // self.tokens_per_image

        selected: List[dict] = []
        used_tokens = 0
        # TODO: CLIP-rank by query similarity. For now: return all unranked.
        for snap_id, info in self._registry.items():
            if snap_id in loaded_ids:
                continue
            if used_tokens + info["tokens"] > self.token_budget:
                continue
            selected.append({"snapshot_id": snap_id, **info})
            used_tokens += info["tokens"]
            if len(selected) >= k:
                break
        return selected
