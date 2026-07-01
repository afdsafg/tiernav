"""Room-View-Object Scene Graph Memory (Contribution 2).

Hierarchical scene memory organizing visual evidence around room-level
structure produced by the room-segmentation algorithm in TSDFPlanner.

Node types:
  - RoomNode: room_id, status, summary, connected_rooms, view/object/evidence ids
  - ViewNode: keyframe/snapshot/panorama view, room_id, position, objects visible
  - ObjectNode: category, instance candidates, room_id, view_ids, confidence, verified
  - EvidenceNode: trajectory evidence id, action, outcome, room_id, key_frames

Query API (Contribution 3 — active memory query):
  query_scene_graph(query, filters) -> dict of candidates
  list_rooms(status) -> list[RoomNode]
  find_objects(category_or_description) -> list[ObjectNode]
  retrieve_evidence(object_id | room_id | trajectory_id) -> list[EvidenceNode]
  mark_rejected(region_id, reason) -> None
  mark_searched(room_id) -> None

The graph is episode-scoped: build a new one per episode. For GOATBench
cross-subtask reuse, persist across subtasks via export/import (V2 feature).
"""
from __future__ import annotations

import logging
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ViewNode:
    view_id: str
    room_id: int
    view_type: str = "keyframe"  # keyframe | panorama | frontier | seed
    position_3d: list = field(default_factory=list)
    objects_visible: list[str] = field(default_factory=list)
    image_path: str = ""
    timestamp: str = ""


@dataclass
class ObjectNode:
    object_id: str
    category: str  # e.g. "oven", "cabinet"
    room_id: int = -1
    view_ids: list[str] = field(default_factory=list)
    position_3d: list = field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False  # confirmed by close inspection
    rejected: bool = False  # inspected and rejected


@dataclass
class EvidenceNode:
    evidence_id: str
    decision_id: int
    action: str
    outcome: str
    room_id: int = -1
    key_frame_ids: list[str] = field(default_factory=list)
    objects_nearby: list[str] = field(default_factory=list)
    progress: str = ""


@dataclass
class RoomNode:
    room_id: int
    status: str = "unexplored"  # unexplored | partially_explored | searched | rejected
    summary: str = ""  # semantic summary, e.g. "kitchen-like room with cabinets"
    connected_rooms: list[int] = field(default_factory=list)
    view_ids: list[str] = field(default_factory=list)
    object_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    visit_count: int = 0
    explore_coverage: float = 0.0  # 0..1, fraction of room explored


class SceneGraphMemory:
    """Room-View-Object hierarchical scene graph memory.

    Fed by:
      - TSDFPlanner.room_regions (room segmentation)
      - Executor TrajectoryEvidence (view + object + evidence nodes)
      - Scene.objects (object detections)

    Queried by:
      - Planner via active memory query (Contribution 3)
    """

    def __init__(self):
        self.rooms: dict[int, RoomNode] = {}
        self.views: dict[str, ViewNode] = {}
        self.objects: dict[str, ObjectNode] = {}
        self.evidence: dict[str, EvidenceNode] = {}
        self._object_id_counter = 0
        self._view_id_counter = 0
        self._evidence_id_counter = 0
        self._query_counter = 0

    # ------------------------------------------------------------------
    # Sync from backend structures
    # ------------------------------------------------------------------

    def sync_rooms_from_tsdf(self, tsdf_planner) -> None:
        """Sync room nodes from TSDFPlanner.room_regions (room segmentation)."""
        room_regions = getattr(tsdf_planner, "room_regions", None) or []
        existing = set(self.rooms.keys())
        seen = set()
        for region in room_regions:
            rid = int(getattr(region, "room_id", -1))
            if rid < 0:
                continue
            seen.add(rid)
            if rid not in self.rooms:
                self.rooms[rid] = RoomNode(room_id=rid)
            # Update connectivity if available
            connected = getattr(region, "connected_rooms", None)
            if connected:
                self.rooms[rid].connected_rooms = sorted(set(int(c) for c in connected))
        # Note: do NOT delete rooms not in seen — they may have been marked rejected

    def add_view(
        self,
        room_id: int,
        view_type: str = "keyframe",
        position_3d: Optional[list] = None,
        objects_visible: Optional[list[str]] = None,
        image_path: str = "",
    ) -> ViewNode:
        """Register a new view node."""
        self._view_id_counter += 1
        view_id = f"view_{self._view_id_counter:03d}"
        view = ViewNode(
            view_id=view_id,
            room_id=room_id,
            view_type=view_type,
            position_3d=position_3d or [],
            objects_visible=objects_visible or [],
            image_path=image_path,
        )
        self.views[view_id] = view
        if room_id in self.rooms:
            if view_id not in self.rooms[room_id].view_ids:
                self.rooms[room_id].view_ids.append(view_id)
            if self.rooms[room_id].status == "unexplored":
                self.rooms[room_id].status = "partially_explored"
        return view

    def add_object(
        self,
        category: str,
        room_id: int = -1,
        view_id: Optional[str] = None,
        position_3d: Optional[list] = None,
        confidence: float = 0.0,
        verified: bool = False,
    ) -> ObjectNode:
        """Register a new object node (or update if same category+room exists)."""
        # Dedup: if an unverified object of same category exists in same room, merge
        for obj in self.objects.values():
            if obj.category.lower() == category.lower() and obj.room_id == room_id and not obj.rejected:
                if view_id and view_id not in obj.view_ids:
                    obj.view_ids.append(view_id)
                if confidence > obj.confidence:
                    obj.confidence = confidence
                if verified:
                    obj.verified = True
                return obj
        self._object_id_counter += 1
        object_id = f"object_{self._object_id_counter:03d}"
        obj = ObjectNode(
            object_id=object_id,
            category=category,
            room_id=room_id,
            view_ids=[view_id] if view_id else [],
            position_3d=position_3d or [],
            confidence=confidence,
            verified=verified,
        )
        self.objects[object_id] = obj
        if room_id in self.rooms:
            if object_id not in self.rooms[room_id].object_ids:
                self.rooms[room_id].object_ids.append(object_id)
        return obj

    def add_evidence(
        self,
        decision_id: int,
        action: str,
        outcome: str,
        room_id: int = -1,
        key_frame_ids: Optional[list[str]] = None,
        objects_nearby: Optional[list[str]] = None,
        progress: str = "",
    ) -> EvidenceNode:
        """Register a trajectory evidence node."""
        self._evidence_id_counter += 1
        evidence_id = f"ev_{self._evidence_id_counter:03d}"
        ev = EvidenceNode(
            evidence_id=evidence_id,
            decision_id=decision_id,
            action=action,
            outcome=outcome,
            room_id=room_id,
            key_frame_ids=key_frame_ids or [],
            objects_nearby=objects_nearby or [],
            progress=progress,
        )
        self.evidence[evidence_id] = ev
        if room_id in self.rooms:
            if evidence_id not in self.rooms[room_id].evidence_ids:
                self.rooms[room_id].evidence_ids.append(evidence_id)
        # Auto-register nearby objects
        for obj_name in objects_nearby or []:
            self.add_object(category=obj_name, room_id=room_id, view_id=None, confidence=0.3)
        return ev

    # ------------------------------------------------------------------
    # Status / mutation
    # ------------------------------------------------------------------

    def mark_rejected(self, room_id: int, reason: str) -> None:
        """Mark a room as rejected (failed search)."""
        if room_id in self.rooms:
            self.rooms[room_id].status = "rejected"
            self.rooms[room_id].rejection_reason = reason

    def mark_searched(self, room_id: int, summary: str = "") -> None:
        """Mark a room as fully searched."""
        if room_id in self.rooms:
            self.rooms[room_id].status = "searched"
            if summary:
                self.rooms[room_id].summary = summary
            self.rooms[room_id].explore_coverage = 1.0

    def mark_object_verified(self, object_id: str, verified: bool = True) -> None:
        if object_id in self.objects:
            self.objects[object_id].verified = verified
            if not verified:
                self.objects[object_id].rejected = True

    def increment_room_visit(self, room_id: int) -> None:
        if room_id in self.rooms:
            self.rooms[room_id].visit_count += 1

    # ------------------------------------------------------------------
    # Query API (Contribution 3 — active memory query)
    # ------------------------------------------------------------------

    def query_scene_graph(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k_rooms: int = 5,
        top_k_views: int = 8,
        top_k_objects: int = 10,
    ) -> dict:
        """Active memory query: room-level prefilter → view/object retrieval.

        Returns a dict with candidate_rooms, candidate_views, candidate_objects,
        returned_evidence_ids. The Planner can then choose to view specific
        evidence via retrieve_evidence.
        """
        t0 = time.time()
        self._query_counter += 1
        filters = filters or {}
        room_statuses = filters.get("status") or [
            "partially_explored", "searched"
        ]
        if isinstance(room_statuses, str):
            room_statuses = [room_statuses]

        # 1. Room-level prefilter: by status + keyword match against summary/object categories
        query_words = set(re.findall(r"\w+", query.lower()))
        candidate_rooms: list[int] = []
        room_scores: dict[int, float] = {}
        for rid, room in self.rooms.items():
            if room.status not in room_statuses:
                continue
            score = 0.0
            # Keyword match on room summary
            summary_words = set(re.findall(r"\w+", room.summary.lower()))
            score += len(query_words & summary_words) * 2.0
            # Keyword match on objects in room
            for oid in room.object_ids:
                obj = self.objects.get(oid)
                if obj and not obj.rejected:
                    cat_words = set(re.findall(r"\w+", obj.category.lower()))
                    score += len(query_words & cat_words) * 1.5
            if score > 0 or not query_words:
                candidate_rooms.append(rid)
                room_scores[rid] = score if query_words else 1.0
        # Sort by score desc, then room_id
        candidate_rooms.sort(key=lambda r: (-room_scores[r], r))
        candidate_rooms = candidate_rooms[:top_k_rooms]

        # 2. View/Object retrieval within candidate rooms
        candidate_views: list[str] = []
        candidate_objects: list[str] = []
        for rid in candidate_rooms:
            room = self.rooms[rid]
            for vid in room.view_ids:
                view = self.views.get(vid)
                if view:
                    # Score by object keyword match
                    v_words = set()
                    for o in view.objects_visible:
                        v_words |= set(re.findall(r"\w+", o.lower()))
                    if query_words & v_words or not query_words:
                        candidate_views.append(vid)
            for oid in room.object_ids:
                obj = self.objects.get(oid)
                if obj and not obj.rejected:
                    candidate_objects.append(oid)
        candidate_views = candidate_views[:top_k_views]
        candidate_objects = candidate_objects[:top_k_objects]

        # 3. Evidence retrieval
        returned_evidence_ids: list[str] = []
        for rid in candidate_rooms:
            returned_evidence_ids.extend(self.rooms[rid].evidence_ids)
        returned_evidence_ids = returned_evidence_ids[:top_k_objects]

        latency = time.time() - t0
        return {
            "query_id": self._query_counter,
            "candidate_rooms": candidate_rooms,
            "candidate_views": candidate_views,
            "candidate_objects": candidate_objects,
            "returned_evidence_ids": returned_evidence_ids,
            "query_latency_sec": latency,
        }

    def list_rooms(self, status: Optional[str | list[str]] = None) -> list[RoomNode]:
        """List rooms, optionally filtered by status."""
        if status is None:
            return list(self.rooms.values())
        statuses = [status] if isinstance(status, str) else status
        return [r for r in self.rooms.values() if r.status in statuses]

    def find_objects(
        self, category_or_description: str, include_rejected: bool = False
    ) -> list[ObjectNode]:
        """Find objects by category keyword match."""
        words = set(re.findall(r"\w+", category_or_description.lower()))
        results = []
        for obj in self.objects.values():
            if obj.rejected and not include_rejected:
                continue
            cat_words = set(re.findall(r"\w+", obj.category.lower()))
            if words & cat_words:
                results.append(obj)
        return results

    def retrieve_evidence(
        self,
        object_id: Optional[str] = None,
        room_id: Optional[int] = None,
        trajectory_id: Optional[str] = None,
    ) -> list[EvidenceNode]:
        """Retrieve evidence nodes by object, room, or trajectory id."""
        results = []
        if trajectory_id and trajectory_id in self.evidence:
            results.append(self.evidence[trajectory_id])
        if room_id is not None:
            for ev in self.evidence.values():
                if ev.room_id == room_id:
                    results.append(ev)
        if object_id:
            obj = self.objects.get(object_id)
            if obj:
                # Evidence from the object's room and views
                for ev in self.evidence.values():
                    if ev.room_id == obj.room_id:
                        results.append(ev)
        # Dedup by evidence_id
        seen = set()
        deduped = []
        for ev in results:
            if ev.evidence_id not in seen:
                seen.add(ev.evidence_id)
                deduped.append(ev)
        return deduped

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full graph for room_view_object_graph.json."""
        return {
            "rooms": [asdict(r) for r in self.rooms.values()],
            "views": [asdict(v) for v in self.views.values()],
            "objects": [asdict(o) for o in self.objects.values()],
            "evidence": [asdict(e) for e in self.evidence.values()],
            "stats": {
                "num_rooms": len(self.rooms),
                "num_views": len(self.views),
                "num_objects": len(self.objects),
                "num_evidence": len(self.evidence),
                "num_queries": self._query_counter,
            },
        }

    def persist_json(self, path: str | os.PathLike) -> str:
        """Persist the materialized scene graph to a local JSON file."""
        out_path = os.fspath(path)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return out_path

    def get_summary_for_planner(self, max_rooms: int = 8) -> str:
        """Compact text summary for injection into the Planner prompt."""
        lines = ["## Scene Graph Memory"]
        if not self.rooms:
            lines.append("- No rooms registered yet.")
            return "\n".join(lines)
        lines.append(f"Total rooms: {len(self.rooms)}")
        for room in sorted(self.rooms.values(), key=lambda r: r.room_id)[:max_rooms]:
            obj_cats = sorted({
                self.objects[oid].category
                for oid in room.object_ids
                if oid in self.objects and not self.objects[oid].rejected
            })
            lines.append(
                f"- Room {room.room_id} [{room.status}] visits={room.visit_count} "
                f"objects={obj_cats or 'none'} evidence={len(room.evidence_ids)}"
            )
        rejected = [r for r in self.rooms.values() if r.status == "rejected"]
        if rejected:
            lines.append(
                f"Rejected rooms: {', '.join(str(r.room_id) + '(' + r.rejection_reason[:40] + ')' for r in rejected)}"
            )
        return "\n".join(lines)
