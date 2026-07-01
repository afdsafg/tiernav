from src.tiernav_runtime.aeqa_predictive import (
    AEQAFrontier,
    AEQAImage,
    AEQAVisualState,
    build_content,
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
