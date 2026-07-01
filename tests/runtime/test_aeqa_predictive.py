from src.tiernav_runtime.aeqa_predictive import (
    AEQAFrontier,
    AEQAImage,
    AEQAVisualState,
    AEQAVisualStateBuilder,
    build_answer_messages,
    build_content,
    build_explore_messages,
    parse_answer_response,
    parse_frontier_response,
    parse_retain_indices,
)


def test_build_content_interleaves_text_and_images():
    content = build_content([
        ("Question: what is on the handle?", None),
        ("Snapshot 0:", "abc123"),
    ])

    assert content == [
        {"type": "text", "text": "Question: what is on the handle?"},
        {"type": "text", "text": "Snapshot 0:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123", "detail": "high"},
        },
    ]


def test_parse_answer_response_extracts_answer_and_evidence():
    parsed = parse_answer_response(
        "I can answer.\nAnswer: a white towel (Evidence: Snapshot 2)"
    )

    assert parsed["action"] == "answer"
    assert parsed["answer"] == "a white towel"
    assert parsed["evidence_snapshot"] == 2


def test_parse_answer_response_continue_exploration():
    parsed = parse_answer_response("Continue Exploration")
    assert parsed == {
        "action": "continue_exploration",
        "answer": "",
        "evidence_snapshot": None,
    }


def test_parse_answer_response_prefers_answer_over_embedded_continue_phrase():
    parsed = parse_answer_response(
        "Reasoning: continue exploration is not needed now.\n"
        "Answer: a white towel (Evidence: Snapshot 1)"
    )

    assert parsed["action"] == "answer"
    assert parsed["answer"] == "a white towel"
    assert parsed["evidence_snapshot"] == 1


def test_parse_answer_response_uses_line_anchored_answer_directive():
    parsed = parse_answer_response(
        "The previous Answer: was wrong.\n"
        "Answer: a blue mug (Evidence: Snapshot 3)"
    )

    assert parsed["action"] == "answer"
    assert parsed["answer"] == "a blue mug"
    assert parsed["evidence_snapshot"] == 3


def test_parse_frontier_response_accepts_next_step_format():
    assert parse_frontier_response("Reasoning...\nNext Step: Frontier 7", ["3", "7"]) == "7"


def test_parse_frontier_response_falls_back_to_first_valid_frontier():
    assert parse_frontier_response("Next Step: Frontier 99", ["3", "7"]) == "3"


def test_parse_retain_indices_accepts_braces_and_filters_bounds():
    assert parse_retain_indices("Retain Snapshots: {0, 2, 9}.", max_count=3) == [0, 2]


def test_visual_state_is_json_serializable():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        current_step=4,
        snapshots=[AEQAImage(image_id="snap-1", image_b64="abc", label="Snapshot 0")],
        frontiers=[AEQAFrontier(frontier_id="5", image_b64="def")],
        egocentric_views=[],
        memory_text="plan line",
        tool_feedback="last action failed",
    )

    payload = state.model_dump(mode="json")
    assert payload["snapshots"][0]["image_id"] == "snap-1"
    assert payload["frontiers"][0]["frontier_id"] == "5"


class FakeVisualEnv:
    def get_aeqa_visual_state(self, episode):
        return {
            "question": episode.prompt,
            "current_step": episode.step_index,
            "snapshots": [
                {"image_id": "snap-0", "image_b64": "aaa", "label": "Snapshot 0"},
            ],
            "frontiers": [
                {"frontier_id": "4", "image_b64": "bbb", "label": "Frontier 4"},
            ],
            "egocentric_views": [],
            "memory_text": "High-Level Plan:\n * find the oven",
            "tool_feedback": "last action: none",
        }


class FakeEpisode:
    prompt = "What is hanging on the oven handle?"
    step_index = 5
    last_observation = None


def test_visual_state_builder_uses_environment_adapter():
    builder = AEQAVisualStateBuilder()
    state = builder.build(FakeEpisode(), FakeVisualEnv())

    assert state.question == "What is hanging on the oven handle?"
    assert state.snapshots[0].image_b64 == "aaa"
    assert state.frontiers[0].frontier_id == "4"
    assert "find the oven" in state.memory_text


def test_build_answer_messages_include_snapshot_image():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[AEQAImage(image_id="snap-0", image_b64="aaa", label="Snapshot 0")],
        frontiers=[],
        egocentric_views=[],
        memory_text="High-Level Plan:\n * inspect oven",
    )

    messages = build_answer_messages(state)

    assert messages[0]["role"] == "system"
    assert "sufficient to answer" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert any(block.get("type") == "image_url" for block in messages[1]["content"])
    assert any("Available Snapshots" in block.get("text", "") for block in messages[1]["content"])
    assert any("Snapshot 0" in block.get("text", "") for block in messages[1]["content"])


def test_build_explore_messages_include_frontier_image():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[],
        frontiers=[AEQAFrontier(frontier_id="4", image_b64="bbb", label="Frontier 4")],
        egocentric_views=[],
    )

    messages = build_explore_messages(state)

    assert messages[0]["role"] == "system"
    assert "PHYSICALLY NAVIGATE" in messages[0]["content"]
    assert any(block.get("type") == "image_url" for block in messages[1]["content"])
    assert any("Frontier 4" in block.get("text", "") for block in messages[1]["content"])


def test_prompt_builders_skip_empty_image_payloads():
    answer_state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[AEQAImage(image_id="snap-empty", image_b64="", label="Ghost Snapshot")],
        frontiers=[],
        egocentric_views=[],
    )
    answer_messages = build_answer_messages(answer_state)

    assert not any("Ghost Snapshot" in block.get("text", "") for block in answer_messages[1]["content"])
    assert any("No snapshots available" in block.get("text", "") for block in answer_messages[1]["content"])

    explore_state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[],
        frontiers=[AEQAFrontier(frontier_id="empty", image_b64="", label="Ghost Frontier")],
        egocentric_views=[],
    )
    explore_messages = build_explore_messages(explore_state)

    assert not any("Ghost Frontier" in block.get("text", "") for block in explore_messages[1]["content"])
    assert any("No frontiers available" in block.get("text", "") for block in explore_messages[1]["content"])
