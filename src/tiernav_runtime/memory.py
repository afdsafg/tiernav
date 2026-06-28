"""Spatial memory service for the TierNav runtime.

Three-layer room-snapshot-object scene graph plus lightweight hypothesis
support/contradiction tracking. Returns context-ready :class:`MemoryPack`
instances that :class:`~tiernav_runtime.context.ContextCompiler` can consume
directly.

Contract-first: depends only on stdlib and the runtime contracts module. No
external services, no LangGraph, no fabricated evidence — evidence_ids are
always snapshot ids derived from real :class:`Observation` image_ids.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from .contracts import ConfidenceScore, MemoryPack, Observation, RuntimeModel


# --- Node models -----------------------------------------------------------


class RoomNode(RuntimeModel):
    """Room-layer node: aggregates snapshot ids observed in a room."""

    room_id: str
    snapshot_ids: list[str] = Field(default_factory=list)


class SnapshotNode(RuntimeModel):
    """Snapshot-layer node: one image_id with its summary and context."""

    snapshot_id: str
    room_id: str
    summary: str = ""
    object_ids: list[str] = Field(default_factory=list)
    round_index: int = 0
    action_type: str = ""


class ObjectNode(RuntimeModel):
    """Object-layer node: aggregates snapshot ids where an object appears."""

    object_id: str
    room_id: str
    snapshot_ids: list[str] = Field(default_factory=list)


class HypothesisNode(RuntimeModel):
    """Hypothesis with accumulated support/contradiction evidence text."""

    hypothesis_id: str
    text: str
    supports: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


# --- Service ---------------------------------------------------------------


def _snapshot_id(image_id: str) -> str:
    """Stable snapshot id derived from a real observation image_id."""
    return f"snap-{image_id}"


class MemoryService:
    """In-memory spatial scene graph + hypothesis tracker.

    When ``enabled=False`` every mutating method is a no-op and :meth:`query`
    returns an empty :class:`MemoryPack`. This lets callers wire the service
    unconditionally and flip it off via ``AblationConfig.spatial_memory``.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled: bool = enabled
        self.rooms: dict[str, RoomNode] = {}
        self.snapshots: dict[str, SnapshotNode] = {}
        self.objects: dict[str, ObjectNode] = {}
        self.hypotheses: dict[str, HypothesisNode] = {}

    # -- mutation -----------------------------------------------------------

    def update_from_observation(
        self,
        observation: Observation,
        action_type: str,
        round_index: int,
    ) -> None:
        """Index an observation into the room/snapshot/object layers.

        - room layer: ``observation.room_id`` or ``"unknown"``.
        - snapshot layer: one node per ``observation.image_id``.
        - object layer: one node per ``observation.object_id``, linked to the
          snapshots that produced it.

        Snapshot ids are never duplicated on repeated updates.
        """
        if not self.enabled:
            return

        room_id = observation.room_id or "unknown"
        room = self.rooms.setdefault(room_id, RoomNode(room_id=room_id))

        new_snapshot_ids: list[str] = []
        for image_id in observation.image_ids:
            snap_id = _snapshot_id(image_id)
            if snap_id not in self.snapshots:
                self.snapshots[snap_id] = SnapshotNode(
                    snapshot_id=snap_id,
                    room_id=room_id,
                    summary=observation.summary,
                    object_ids=list(observation.object_ids),
                    round_index=round_index,
                    action_type=action_type,
                )
                new_snapshot_ids.append(snap_id)
            if snap_id not in room.snapshot_ids:
                room.snapshot_ids.append(snap_id)

        for object_id in observation.object_ids:
            obj = self.objects.setdefault(
                object_id, ObjectNode(object_id=object_id, room_id=room_id)
            )
            for snap_id in new_snapshot_ids:
                if snap_id not in obj.snapshot_ids:
                    obj.snapshot_ids.append(snap_id)

    def add_hypothesis(self, hypothesis_id: str, text: str) -> None:
        if not self.enabled:
            return
        self.hypotheses.setdefault(
            hypothesis_id, HypothesisNode(hypothesis_id=hypothesis_id, text=text)
        )

    def support_hypothesis(self, hypothesis_id: str, evidence: str) -> None:
        if not self.enabled:
            return
        hyp = self.hypotheses.get(hypothesis_id)
        if hyp is not None and evidence not in hyp.supports:
            hyp.supports.append(evidence)

    def contradict_hypothesis(self, hypothesis_id: str, evidence: str) -> None:
        if not self.enabled:
            return
        hyp = self.hypotheses.get(hypothesis_id)
        if hyp is not None and evidence not in hyp.contradictions:
            hyp.contradictions.append(evidence)

    # -- query --------------------------------------------------------------

    def query(self, query: str) -> MemoryPack:
        """Return a context-ready :class:`MemoryPack`.

        Retrieval is deterministic keyword matching: a snapshot matches if any
        whitespace token of ``query`` appears in its summary (case-insensitive).
        On no direct match but existing snapshots, fall back to the first few
        snapshots so context stays reusable. Aggregates all hypothesis
        supports/contradictions.
        """
        if not self.enabled:
            return MemoryPack(query=query, summary="")

        matched: list[SnapshotNode] = []
        if query.strip():
            tokens = {t.lower() for t in query.split() if t}
            for snap in self.snapshots.values():
                summary_tokens = {s.lower() for s in snap.summary.split() if s}
                if tokens & summary_tokens:
                    matched.append(snap)

        if not matched and self.snapshots:
            # fallback: first few existing snapshots, keep context reusable.
            matched = list(self.snapshots.values())[:3]

        if not matched:
            return MemoryPack(query=query, summary="")

        summary = " | ".join(s.summary for s in matched if s.summary)
        evidence_ids = [s.snapshot_id for s in matched]
        reuse_hint = self._reuse_hint(matched)

        supports: list[str] = []
        contradictions: list[str] = []
        for hyp in self.hypotheses.values():
            supports.extend(hyp.supports)
            contradictions.extend(hyp.contradictions)

        confidence: ConfidenceScore = 1.0 if matched else 0.0

        return MemoryPack(
            query=query,
            summary=summary,
            evidence_ids=evidence_ids,
            supports=supports,
            contradictions=contradictions,
            confidence=confidence,
            reuse_hint=reuse_hint,
        )

    @staticmethod
    def _reuse_hint(matched: list[SnapshotNode]) -> str:
        if not matched:
            return ""
        room_ids: list[str] = []
        for s in matched:
            if s.room_id not in room_ids:
                room_ids.append(s.room_id)
        rooms = ", ".join(room_ids)
        return f"reuse snapshots from room(s): {rooms}" if rooms else "reuse existing snapshots"
