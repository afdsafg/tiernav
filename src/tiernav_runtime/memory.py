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

from typing import Any, Callable, Optional

from pydantic import Field

from .contracts import ConfidenceScore, MemoryPack, MemoryScope, Observation, RuntimeModel


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
        snapshots so context stays reusable — but downgrade ``confidence`` to
        0.0 so downstream consumers can distinguish this best-effort reuse from
        a real hit. Aggregates all hypothesis supports/contradictions.
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

        is_fallback = False
        if not matched and self.snapshots:
            # fallback: first few existing snapshots, keep context reusable.
            # Marked as fallback so confidence is downgraded and the hint
            # signals to downstream that this is not a direct hit.
            is_fallback = True
            matched = list(self.snapshots.values())[:3]

        if not matched:
            return MemoryPack(query=query, summary="")

        summary = " | ".join(s.summary for s in matched if s.summary)
        evidence_ids = [s.snapshot_id for s in matched]
        reuse_hint = self._reuse_hint(matched, is_fallback=is_fallback)

        supports: list[str] = []
        contradictions: list[str] = []
        for hyp in self.hypotheses.values():
            supports.extend(hyp.supports)
            contradictions.extend(hyp.contradictions)

        # Only a direct keyword match justifies high confidence. Fallback reuse
        # returns real snapshot ids but is not evidence the query was answered.
        confidence: ConfidenceScore = 0.0 if is_fallback else 1.0

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
    def _reuse_hint(matched: list[SnapshotNode], *, is_fallback: bool = False) -> str:
        if not matched:
            return ""
        room_ids: list[str] = []
        for s in matched:
            if s.room_id not in room_ids:
                room_ids.append(s.room_id)
        rooms = ", ".join(room_ids)
        if is_fallback:
            base = "fallback reuse of existing snapshots"
            return f"{base} from room(s): {rooms}" if rooms else base
        return f"reuse snapshots from room(s): {rooms}" if rooms else "reuse existing snapshots"


# --- Session ---------------------------------------------------------------


class MemorySession:
    """Episode-scoped ownership of a :class:`MemoryService`.

    Encapsulates the benchmark-dependent lifetime of spatial memory:

    - ``MemoryScope.PER_QUESTION`` (AEQA): each ``start_session`` creates a
      fresh :class:`MemoryService`. Nothing carries over between questions,
      even within the same episode — each question is scored independently.
    - ``MemoryScope.SUBTASK_SEQUENCE`` (GOATBench): the same
      :class:`MemoryService` (and the injected ``notebook`` /
      ``scene_graph``) is reused across subtasks of one episode; a new
      ``episode_id`` resets the memory.

    The session also bridges real observations into the active memory layer by
    delegating :meth:`update_from_observation` and :meth:`query` to the active
    :class:`MemoryService`, so callers (e.g. the graph) can talk to the
    session as if it were the service.

    ``notebook`` and ``scene_graph`` are duck-typed and owned by the session
    for the duration of the session object's lifetime. They are only meaningfully
    populated for ``SUBTASK_SEQUENCE``; for ``PER_QUESTION`` they remain
    ``None`` (AEQA has no cross-question reuse). Task 8 (adapters) injects the
    real ``Notebook`` / ``SceneGraphMemory`` instances.
    """

    def __init__(
        self,
        scope: MemoryScope,
        memory_factory: Callable[..., MemoryService] = MemoryService,
        notebook: Optional[Any] = None,
        scene_graph: Optional[Any] = None,
        scene_graph_source: Optional[Any] = None,
    ) -> None:
        self.scope: MemoryScope = scope
        self._memory_factory = memory_factory
        # notebook/scene_graph only meaningful for SUBTASK_SEQUENCE.
        self.notebook: Optional[Any] = notebook if scope is MemoryScope.SUBTASK_SEQUENCE else None
        self.scene_graph: Optional[Any] = (
            scene_graph if scope is MemoryScope.SUBTASK_SEQUENCE else None
        )
        self.scene_graph_source: Optional[Any] = (
            scene_graph_source if scope is MemoryScope.SUBTASK_SEQUENCE else None
        )
        self._active: Optional[MemoryService] = None
        self._episode_id: Optional[str] = None

    @property
    def current_memory(self) -> MemoryService:
        """The active :class:`MemoryService` for the current session.

        Raises if called before :meth:`start_session` — the graph must open a
        session before reading or writing memory.
        """
        if self._active is None:
            raise RuntimeError(
                "MemorySession.start_session must be called before accessing memory"
            )
        return self._active

    def start_session(
        self,
        episode_id: str,
        *,
        question_id: Optional[str] = None,
        subtask_index: Optional[int] = None,
    ) -> MemoryService:
        """Open a memory session for one question or subtask.

        - ``PER_QUESTION``: always create a fresh :class:`MemoryService`.
        - ``SUBTASK_SEQUENCE``: reuse the existing service if ``episode_id``
          matches the previous call; otherwise create a fresh one.

        Returns the active :class:`MemoryService`.
        """
        if self.scope is MemoryScope.PER_QUESTION:
            self._active = self._memory_factory()
            self._episode_id = episode_id
            return self._active

        # SUBTASK_SEQUENCE: persist across subtasks within the same episode.
        if self._active is not None and episode_id == self._episode_id:
            return self._active
        self._active = self._memory_factory()
        self._episode_id = episode_id
        return self._active

    # -- delegation: bridge observations into the active service ------------

    def update_from_observation(
        self,
        observation: Observation,
        action_type: str,
        round_index: int,
    ) -> None:
        """Bridge a real observation into the active :class:`MemoryService`."""
        self.current_memory.update_from_observation(
            observation, action_type=action_type, round_index=round_index
        )
        self._mirror_scene_graph_observation(
            observation, action_type=action_type, round_index=round_index
        )

    def query(self, query: str) -> MemoryPack:
        """Query the active :class:`MemoryService`."""
        return self.current_memory.query(query)

    def _mirror_scene_graph_observation(
        self,
        observation: Observation,
        *,
        action_type: str,
        round_index: int,
    ) -> None:
        """Mirror runtime observations into optional SceneGraphMemory.

        This is intentionally duck-typed so the runtime layer does not import
        the heavier scene graph module at import time. Failures are swallowed:
        the contract memory remains the source used by the graph, and the
        local scene-graph artifact is a best-effort materialized view.
        """
        graph = self.scene_graph
        if graph is None or not hasattr(graph, "add_evidence"):
            return

        room_id = _room_id_as_int(observation.room_id)
        summary = observation.summary or str(observation.raw.get("progress", "") or "")
        outcome = str(observation.raw.get("outcome", "") or summary)
        try:
            if (
                self.scene_graph_source is not None
                and hasattr(graph, "sync_rooms_from_tsdf")
            ):
                graph.sync_rooms_from_tsdf(self.scene_graph_source)
            graph.add_evidence(
                decision_id=int(round_index),
                action=action_type,
                outcome=outcome,
                room_id=room_id,
                key_frame_ids=list(observation.image_ids),
                objects_nearby=list(observation.object_ids),
                progress=summary,
            )
            if room_id >= 0 and hasattr(graph, "increment_room_visit"):
                graph.increment_room_visit(room_id)
        except Exception:
            return


def _room_id_as_int(room_id: Optional[str]) -> int:
    if room_id is None:
        return -1
    try:
        return int(room_id)
    except (TypeError, ValueError):
        return -1
