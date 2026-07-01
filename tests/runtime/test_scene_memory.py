"""Tests for SceneMemoryStore: cross-episode scene memory with JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tiernav_runtime.scene_memory import SceneMemoryStore


class FakePlanner:
    """Minimal planner_client stub: call_vlm returns a fixed string."""

    def __init__(self, raw: str) -> None:
        self._raw = raw
        self.captured_messages: list = []

    def call_vlm(self, messages, **kwargs):
        self.captured_messages = messages
        return self._raw


def test_update_room_and_manifest(tmp_path):
    store = SceneMemoryStore("scene-1", str(tmp_path))
    store.update_room("0", ["chair", "table"], visited_round=0, connectivity=["1"], notes="kitchen")
    store.update_room("0", ["fridge"], visited_round=2, connectivity=["1", "2"])
    store.update_room("1", ["bed"], visited_round=1)

    manifest = store.get_manifest()
    assert "0(visited)" in manifest
    assert "1(visited)" in manifest
    assert "chair x1" in manifest
    assert "table x1" in manifest
    assert "fridge x1" in manifest

    room0 = store.data["rooms"]["0"]
    assert room0["status"] == "visited"
    assert room0["visit_count"] == 2
    assert set(room0["objects_seen"]) == {"chair", "table", "fridge"}
    assert set(room0["connected_rooms"]) == {"1", "2"}
    assert room0["notes"] == "kitchen"


def test_add_episodic_note_and_manifest_count(tmp_path):
    store = SceneMemoryStore("scene-2", str(tmp_path))
    store.add_episodic_note(round=0, room="0", event="found fridge")
    store.add_episodic_note(round=1, room="1", event="saw bed")
    store.add_episodic_note(round=3, room="0", event="back to kitchen")

    manifest = store.get_manifest()
    assert "3 entries" in manifest
    assert "rounds 0, 1, 3" in manifest
    assert len(store.data["episodic_notes"]) == 3


def test_get_node_detail_room(tmp_path):
    store = SceneMemoryStore("scene-3", str(tmp_path))
    store.update_room("2", ["desk", "lamp"], visited_round=0, notes="office")
    detail = store.get_node_detail("room", "2")
    assert detail["room_id"] == "2"
    assert detail["status"] == "visited"
    assert "desk" in detail["objects_seen"]
    assert detail["notes"] == "office"

    assert store.get_node_detail("room", "999") == {}


def test_get_node_detail_object(tmp_path):
    store = SceneMemoryStore("scene-4", str(tmp_path))
    store.update_room("0", ["chair"], visited_round=0)
    store.update_room("1", ["chair", "bed"], visited_round=1)
    store.add_episodic_note(round=0, room="0", event="chair near table")
    detail = store.get_node_detail("object", "chair")
    assert set(detail["rooms"]) == {"0", "1"}
    assert len(detail["related_notes"]) == 1


def test_recall_with_mock_planner(tmp_path):
    store = SceneMemoryStore("scene-5", str(tmp_path))
    store.update_room("0", ["fridge"], visited_round=0)
    store.update_room("1", ["bed"], visited_round=1)
    manifest = store.get_manifest()

    raw = json.dumps({
        "recall": [
            {"type": "room", "id": "0", "reason": "fridge likely has target"},
            {"type": "object", "id": "fridge", "reason": "target may be food"},
        ]
    })
    planner = FakePlanner(raw)
    result = store.recall("find food", manifest, "1", planner)

    assert len(result) == 2
    assert result[0] == {"type": "room", "id": "0", "reason": "fridge likely has target"}
    assert result[1]["type"] == "object"
    assert result[1]["id"] == "fridge"
    # prompt reached the planner
    assert "find food" in planner.captured_messages[0]["content"]
    assert manifest in planner.captured_messages[0]["content"]


def test_save_load_round_trip(tmp_path):
    store = SceneMemoryStore("scene-6", str(tmp_path))
    store.update_room("0", ["chair"], visited_round=0, notes="kitchen")
    store.add_episodic_note(round=0, room="0", event="entered kitchen")
    pkl_path = store.path
    assert pkl_path.exists()

    store2 = SceneMemoryStore("scene-6", str(tmp_path))
    assert store2.data["scene_id"] == "scene-6"
    assert store2.data["rooms"]["0"]["objects_seen"] == ["chair"]
    assert store2.data["rooms"]["0"]["notes"] == "kitchen"
    assert len(store2.data["episodic_notes"]) == 1
    assert store2.data["episodic_notes"][0]["event"] == "entered kitchen"
    assert store2.data["last_updated"]  # persisted


def test_recall_parse_failure_returns_empty(tmp_path):
    store = SceneMemoryStore("scene-7", str(tmp_path))
    store.update_room("0", ["chair"], visited_round=0)

    # garbage JSON
    planner = FakePlanner("not json at all")
    assert store.recall("goal", store.get_manifest(), "0", planner) == []

    # missing 'recall' key
    planner2 = FakePlanner(json.dumps({"other": []}))
    assert store.recall("goal", store.get_manifest(), "0", planner2) == []

    # empty string
    planner3 = FakePlanner("")
    assert store.recall("goal", store.get_manifest(), "0", planner3) == []


def test_recall_planner_exception_returns_empty(tmp_path):
    store = SceneMemoryStore("scene-8", str(tmp_path))
    store.update_room("0", ["chair"], visited_round=0)

    class Boom:
        def call_vlm(self, messages, **kwargs):
            raise RuntimeError("network down")

    assert store.recall("goal", store.get_manifest(), "0", Boom()) == []
