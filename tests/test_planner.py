"""Tests for src/agent_planner.py — PlannerAction, prompt building, JSON parsing."""
from __future__ import annotations

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------
from src.agent_planner import PLANNER_SYSTEM_PROMPT, Planner, PlannerAction
from src.agent_workflow import _build_frontier_mosaic_b64, _parse_stage65_frontier_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def planner():
    return Planner(api_key="test-key", base_url="https://example.invalid/v1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_parse_planner_response(planner):
    response = (
        '{"reason": "target item is visible in the cited snapshot", '
        '"expected": "navigation should bring the agent closer to the cited target", '
        '"action": "navigate_to_object", '
        '"arguments": {"snapshot_id": "step12_view1", "object_name": "target item"}, '
        '"confidence": 0.7}'
    )
    action = planner.parse_response(response)
    assert action.action_type == "navigate_to_object"
    assert action.snapshot_id == "step12_view1"
    assert action.object_name == "target item"
    assert action.expected == "navigation should bring the agent closer to the cited target"
    assert pytest.approx(action.confidence) == 0.7
    assert action.reason == "target item is visible in the cited snapshot"


def test_parse_planner_response_with_markdown(planner):
    response = "```json\n{\n  \"action\": \"explore_seed\",\n  \"seed_id\": 3,\n  \"confidence\": 0.85\n}\n```"
    action = planner.parse_response(response)
    assert action.action_type == "explore_seed"
    assert action.seed_id == "3"
    assert pytest.approx(action.confidence) == 0.85


def test_parse_planner_response_bad_json_fallback(planner):
    action = planner.parse_response("This is not JSON at all.")
    assert action.action_type == "explore_panorama"
    assert "Parse failed" in action.reason


def test_parse_non_json_submit_answer_does_not_submit(planner):
    action = planner.parse_response("* Let's formulate the response.\nsubmit_answer")
    assert action.action_type == "explore_panorama"
    assert action.answer is None
    assert "Invalid non-JSON submit_answer" in action.reason


def test_parse_submit_answer_requires_answer(planner):
    action = planner.parse_response(
        '{"action": "submit_answer", "arguments": {"snapshot_id": "step1_view1"}}'
    )
    assert action.action_type == "explore_panorama"
    assert action.answer is None
    assert "missing answer" in action.reason


def test_build_prompt_contains_components(planner):
    history = "## History\n- [Step 5] Room explored; target not found"
    scene = "## Scene Analysis\n- View 0: [furniture, doorway]"
    progress = "## Progress\nTarget not found. Continue exploring."
    actions = "## Actions\n1. navigate_to_object\n2. explore_seed"

    prompt = planner.build_prompt(
        question="What object is on the referenced surface?",
        history=history,
        scene=scene,
        progress=progress,
        actions=actions,
    )
    assert "History" in prompt
    assert "Scene Analysis" in prompt
    assert "Progress" in prompt
    assert "What object is on the referenced surface?" in prompt


def test_system_prompt_prioritizes_submit_answer():
    assert "submit_answer immediately" in PLANNER_SYSTEM_PROMPT
    assert "current-view snapshots" in PLANNER_SYSTEM_PROMPT
    assert "snapshot_id supporting the answer" in PLANNER_SYSTEM_PROMPT
    assert "Exploration/navigation actions must include expected" in PLANNER_SYSTEM_PROMPT
    assert "submit_answer is terminal and does not need expected" in PLANNER_SYSTEM_PROMPT


def test_decide_sends_three_current_images_and_topdown(planner, monkeypatch):
    captured = {}

    def fake_call_api(messages):
        captured["messages"] = messages
        return (
            '{"action": "submit_answer", '
            '"arguments": {"snapshot_id": "step4_view1", "answer": "done"}, '
            '"confidence": 1.0}'
        )

    monkeypatch.setattr(planner, "_call_api", fake_call_api)
    action = planner.decide(
        question="q",
        history="## History",
        scene="## Scene",
        progress="## Progress",
        actions="## Actions",
        image_b64s=["left-view", "front-view", "right-view", "topdown-map"],
    )

    assert action.action_type == "submit_answer"
    assert action.snapshot_id == "step4_view1"
    assert action.expected is None
    content = captured["messages"][-1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 4
    assert image_parts[0]["image_url"]["url"].endswith("left-view")
    assert image_parts[1]["image_url"]["url"].endswith("front-view")
    assert image_parts[2]["image_url"]["url"].endswith("right-view")
    assert image_parts[3]["image_url"]["url"].endswith("topdown-map")


def test_planner_action_dataclass():
    action = PlannerAction(
        action_type="explore_seed",
        seed_id="3",
        confidence=0.6,
        reason="seed 3 leads to another room",
    )
    assert action.action_type == "explore_seed"
    assert action.seed_id == "3"
    assert pytest.approx(action.confidence) == 0.6
    assert action.reason == "seed 3 leads to another room"
    assert action.object_name is None
    assert action.snapshot_id is None
    assert action.frontier_id is None
    assert action.answer is None
    assert action.expected is None


def test_planner_action_submit_answer():
    action = PlannerAction(
        action_type="submit_answer",
        answer="a visible item",
        confidence=1.0,
        reason="answer is supported by the cited snapshot",
    )
    assert action.action_type == "submit_answer"
    assert action.answer == "a visible item"
    assert action.expected is None


def test_planner_action_defaults():
    action = PlannerAction(action_type="explore_panorama")
    assert action.action_type == "explore_panorama"
    assert action.reason == ""
    assert action.confidence == 0.0
    assert action.object_name is None
    assert action.snapshot_id is None
    assert action.seed_id is None
    assert action.frontier_id is None
    assert action.view_idx is None
    assert action.answer is None
    assert action.expected is None


def test_build_frontier_mosaic_b64_labels_ids():
    class Frontier:
        def __init__(self, frontier_id):
            self.frontier_id = frontier_id
            self.feature = np.zeros((32, 32, 3), dtype=np.uint8)

    b64, ids = _build_frontier_mosaic_b64([Frontier(7), Frontier(9)])
    assert b64
    assert ids == [7, 9]


def test_parse_stage65_frontier_response_from_text():
    response = "Frontier 7 shows a clear kitchen-like area, so it is the best next target."
    parsed = _parse_stage65_frontier_response(response, [7, 9, 11])
    assert parsed["frontier_id"] == 7
    assert "kitchen-like" in parsed["reasoning"]
