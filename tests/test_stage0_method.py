"""Smoke tests for the new Stage 0 + method modules."""
import json
import os
import tempfile

from src.run_logger import RunLogger
from src.scene_graph_memory import SceneGraphMemory, RoomNode
from src.agent_notebook import EvidenceNotebook, StructuredNotebook


def test_run_logger_lifecycle():
    """RunLogger should produce all trace files with correct schema."""
    with tempfile.TemporaryDirectory() as tmp:
        rl = RunLogger(
            output_dir=tmp,
            run_id="test_run",
            method_name="ours_full",
            dataset="AEQA-41",
            model="qwen2.5vl-3b",
            seed=42,
            config_path="cfg/eval_aeqa.yaml",
        )
        rl.start_episode("q1", "What color is the sofa?")
        rl.log_decision(
            episode_id="q1", decision_id=1, current_room=2,
            notebook_before={}, available_actions=["explore_panorama", "submit_answer"],
            planner_reason="Need to find sofa", selected_action="explore_panorama",
            target="", expected_evidence="sofa visible",
        )
        rl.log_memory_query(
            episode_id="q1", decision_id=1, query_id=1,
            query_text="rooms with sofa", candidate_rooms=[2],
            returned_evidence_ids=["ev_001"], evidence_viewed_by_planner=["ev_001"],
        )
        rl.log_trajectory_evidence(
            episode_id="q1", decision_id=1, action="explore_panorama",
            target="", outcome="panorama_complete", room_id=2,
            objects_nearby=["sofa", "lamp"], key_frame_ids=["ev_001"],
        )
        rl.save_graph("q1", {"rooms": [], "views": [], "objects": [], "evidence": []})
        rl.finalize_episode(
            episode_id="q1", success=True, answer="blue",
            evidence_ids=["ev_001"], path_length=3.5, num_steps=10,
        )
        rl.close()

        # Verify all files exist
        for fname in [
            "run_manifest.json", "episode_metrics.csv", "decision_trace.jsonl",
            "memory_query_trace.jsonl", "trajectory_evidence.jsonl",
            "answer_evidence.json", "room_view_object_graph.json", "failures.csv",
        ]:
            assert os.path.exists(os.path.join(tmp, fname)), f"missing {fname}"

        # Verify manifest
        with open(os.path.join(tmp, "run_manifest.json")) as fh:
            m = json.load(fh)
        assert m["run_id"] == "test_run"
        assert m["method_name"] == "ours_full"
        assert m["total_episodes"] == 1
        assert m["total_decisions"] == 1
        assert m["total_memory_queries"] == 1
        assert m["end_time"] != ""

        # Verify decision trace
        with open(os.path.join(tmp, "decision_trace.jsonl")) as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["selected_action"] == "explore_panorama"

        # Verify metrics
        with open(os.path.join(tmp, "episode_metrics.csv")) as fh:
            content = fh.read()
        assert "q1" in content
        assert "blue" not in content  # answer is in answer_evidence.json, not csv
        assert "1" in content  # success=1


def test_scene_graph_memory_query():
    """SceneGraphMemory query API should filter by status and keywords."""
    sgm = SceneGraphMemory()
    # Manually add rooms (normally synced from tsdf)
    sgm.rooms[1] = RoomNode(room_id=1, status="searched", summary="living room with sofa")
    sgm.rooms[2] = RoomNode(room_id=2, status="partially_explored", summary="kitchen with oven")
    sgm.rooms[3] = RoomNode(room_id=3, status="unexplored", summary="bedroom")

    sgm.add_object(category="sofa", room_id=1, confidence=0.9, verified=True)
    sgm.add_object(category="oven", room_id=2, confidence=0.7)
    sgm.add_object(category="bed", room_id=3, confidence=0.5)

    # Query for kitchen-related rooms
    result = sgm.query_scene_graph("kitchen oven", filters={"status": ["partially_explored", "searched"]})
    assert 2 in result["candidate_rooms"]
    assert 1 not in result["candidate_rooms"]  # living room doesn't match kitchen/oven
    assert 3 not in result["candidate_rooms"]  # unexplored, filtered out

    # find_objects
    objs = sgm.find_objects("sofa")
    assert len(objs) == 1
    assert objs[0].category == "sofa"

    # mark rejected
    sgm.mark_rejected(2, "no appliance found")
    assert sgm.rooms[2].status == "rejected"
    # Rejected room should not appear in default query
    result2 = sgm.query_scene_graph("oven", filters={"status": ["partially_explored", "searched"]})
    assert 2 not in result2["candidate_rooms"]


def test_scene_graph_to_dict():
    sgm = SceneGraphMemory()
    sgm.rooms[1] = RoomNode(room_id=1, status="searched")
    sgm.add_view(room_id=1, objects_visible=["table"])
    sgm.add_object(category="table", room_id=1, confidence=0.8)
    sgm.add_evidence(decision_id=1, action="explore_panorama", outcome="panorama_complete", room_id=1)
    d = sgm.to_dict()
    assert d["stats"]["num_rooms"] == 1
    assert d["stats"]["num_views"] == 1
    assert d["stats"]["num_objects"] == 1
    assert d["stats"]["num_evidence"] == 1


def test_structured_notebook():
    nb = EvidenceNotebook()
    # Apply a planner update
    nb.apply_planner_update({
        "hypotheses": ["target is in kitchen"],
        "todo": ["check oven area"],
        "rejected": [{"region_id": "room_3", "reason": "no appliances"}],
        "evidence_ids": ["ev_001"],
    })
    assert "target is in kitchen" in nb.structured.hypotheses
    assert "check oven area" in nb.structured.todo
    assert nb.structured.is_rejected("room_3")
    assert "ev_001" in nb.structured.evidence_ids

    # to_dict for decision trace
    d = nb.to_dict()
    assert "structured" in d
    assert d["structured"]["hypotheses"] == ["target is in kitchen"]

    # Injection text
    text = nb.structured.get_injection_text()
    assert "Hypotheses" in text
    assert "Rejected regions" in text
