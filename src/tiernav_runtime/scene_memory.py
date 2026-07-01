"""Cross-episode scene-level structured memory with JSON persistence.

Persists room/object observations and episodic notes to
``scene_memory/<scene_id>.json`` so later episodes in the same scene can
recall what earlier episodes learned. Recall is driven by the same planner
model the graph uses (via ``planner_client.call_vlm``).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SceneMemoryStore:
    """跨 episode 的 scene 级结构化记忆，JSON 持久化。"""

    def __init__(self, scene_id: str, output_dir: str) -> None:
        self.scene_id = scene_id
        self.path = Path(output_dir) / "scene_memory" / (scene_id + ".json")
        self.data: dict = {
            "scene_id": scene_id,
            "rooms": {},
            "episodic_notes": [],
            "last_updated": "",
        }
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        """Read JSON from disk; no-op if file does not exist."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                self.data = loaded
        except (OSError, json.JSONDecodeError):
            logger.warning("scene_memory: failed to load %s, starting fresh", self.path)

    def _save(self) -> None:
        """Incremental write: serialize self.data to self.path."""
        self.data["last_updated"] = datetime.now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    # -- mutation -----------------------------------------------------------

    def update_room(
        self,
        room_id: str,
        objects_seen: list[str],
        visited_round: int,
        connectivity: list[str] | None = None,
        notes: str = "",
    ) -> None:
        """Upsert a room node; merge new objects, bump visit_count, update notes."""
        room = self.data["rooms"].setdefault(
            room_id,
            {
                "room_id": room_id,
                "status": "unvisited",
                "objects_seen": [],
                "visit_count": 0,
                "connected_rooms": [],
                "notes": "",
            },
        )
        room["status"] = "visited"
        for obj in objects_seen:
            if obj not in room["objects_seen"]:
                room["objects_seen"].append(obj)
        room["visit_count"] += 1
        if connectivity is not None:
            for c in connectivity:
                if c not in room["connected_rooms"]:
                    room["connected_rooms"].append(c)
        if notes:
            room["notes"] = notes
        self._save()

    def add_episodic_note(self, round: int, room: str, event: str) -> None:
        """Append an episodic note and persist."""
        self.data["episodic_notes"].append(
            {"round": round, "room": room, "event": event}
        )
        self._save()

    # -- query --------------------------------------------------------------

    def get_manifest(self) -> str:
        """Return structural summary (no details): rooms, objects count, notes count."""
        rooms_visited: list[str] = []
        rooms_unvisited: list[str] = []
        object_counts: dict[str, int] = {}
        for rid, room in self.data["rooms"].items():
            if room.get("status") == "visited":
                rooms_visited.append(str(rid))
            else:
                rooms_unvisited.append(str(rid))
            for obj in room.get("objects_seen", []):
                object_counts[obj] = object_counts.get(obj, 0) + 1

        parts: list[str] = []
        room_parts: list[str] = []
        for rid in rooms_visited:
            room_parts.append(rid + "(visited)")
        for rid in rooms_unvisited:
            room_parts.append(rid + "(unvisited)")
        parts.append("rooms: " + ", ".join(room_parts))

        obj_parts: list[str] = []
        for obj in sorted(object_counts):
            obj_parts.append(obj + " x" + str(object_counts[obj]))
        parts.append("objects: " + (", ".join(obj_parts) if obj_parts else "none"))

        notes = self.data["episodic_notes"]
        rounds = sorted({n["round"] for n in notes})
        rounds_str = ", ".join(str(r) for r in rounds) if rounds else ""
        notes_line = str(len(notes)) + " entries"
        if rounds_str:
            notes_line += " (rounds " + rounds_str + ")"
        parts.append("episodic_notes: " + notes_line)

        return "\n".join(parts)

    def get_node_detail(self, node_type: str, node_id: str) -> dict:
        """Return detail dict for a room or object node. Empty dict if not found."""
        if node_type == "room":
            room = self.data["rooms"].get(node_id)
            return dict(room) if room is not None else {}
        if node_type == "object":
            # Aggregate rooms where object appears + snapshots/notes context.
            rooms_with: list[str] = []
            for rid, room in self.data["rooms"].items():
                if node_id in room.get("objects_seen", []):
                    rooms_with.append(rid)
            notes_with: list[dict] = [
                n for n in self.data["episodic_notes"]
                if node_id in n.get("event", "")
            ]
            return {
                "object_id": node_id,
                "rooms": rooms_with,
                "related_notes": notes_with,
            }
        return {}

    def recall(
        self,
        query: str,
        manifest: str,
        current_room: str,
        planner_client: Any,
    ) -> list[dict]:
        """Ask the planner model which nodes to recall.

        Returns a list of ``{type, id, reason}`` dicts. On parse failure or
        empty response, returns an empty list (never raises).
        """
        prompt = (
            "You are recalling scene memory to help with a navigation task.\n"
            "Goal: " + query + "\n"
            "Current room: " + current_room + "\n"
            "Memory manifest:\n" + manifest + "\n\n"
            "Decide which memory nodes to recall. Output ONLY JSON:\n"
            '{"recall": [{"type": "room", "id": "<room_id>", "reason": "..."}, '
            '{"type": "object", "id": "<object_name>", "reason": "..."}]}'
        )
        try:
            raw = planner_client.call_vlm([{"role": "user", "content": prompt}])
        except Exception:
            logger.warning("scene_memory.recall: planner call failed", exc_info=True)
            return []

        if not raw or not raw.strip():
            return []
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning("scene_memory.recall: JSON parse failed: %.200s", raw)
            return []

        items = parsed.get("recall") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            return []

        out: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            i = item.get("id")
            if t is None or i is None:
                continue
            out.append({"type": str(t), "id": str(i), "reason": str(item.get("reason", ""))})
        return out
