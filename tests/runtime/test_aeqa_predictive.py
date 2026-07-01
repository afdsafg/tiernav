from src.tiernav_runtime.aeqa_predictive import (
    AEQAFrontier,
    AEQAImage,
    AEQAPredictiveController,
    AEQAVisualState,
    AEQAVisualStateBuilder,
    build_answer_messages,
    build_content,
    build_explore_messages,
    parse_answer_response,
    parse_frontier_response,
    parse_retain_indices,
)
from src.tiernav_runtime.contracts import EpisodeState


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


def test_build_answer_messages_include_egocentric_image_when_snapshots_absent():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[],
        frontiers=[],
        egocentric_views=[
            AEQAImage(
                image_id="current_view",
                image_b64="current-b64",
                label="Current egocentric view",
                source="egocentric",
            )
        ],
    )

    messages = build_answer_messages(state)
    content = messages[1]["content"]

    assert any("Current / Egocentric Views" in block.get("text", "") for block in content)
    assert any("Current egocentric view" in block.get("text", "") for block in content)
    assert any(
        block.get("image_url", {}).get("url", "").endswith(",current-b64")
        for block in content
        if block.get("type") == "image_url"
    )


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


def test_build_explore_messages_include_egocentric_clues_with_frontiers():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[],
        frontiers=[AEQAFrontier(frontier_id="4", image_b64="frontier-b64", label="Frontier 4")],
        egocentric_views=[
            AEQAImage(
                image_id="current_view",
                image_b64="current-b64",
                label="Current egocentric view",
                source="egocentric",
            )
        ],
    )

    messages = build_explore_messages(state)
    content = messages[1]["content"]

    assert any("Current / Egocentric Views" in block.get("text", "") for block in content)
    assert any("Current egocentric view" in block.get("text", "") for block in content)
    image_urls = [
        block["image_url"]["url"]
        for block in content
        if block.get("type") == "image_url"
    ]
    assert any(url.endswith(",current-b64") for url in image_urls)
    assert any(url.endswith(",frontier-b64") for url in image_urls)


def test_build_explore_messages_groups_visual_clues_before_frontiers():
    state = AEQAVisualState(
        question="What is hanging on the oven handle?",
        snapshots=[AEQAImage(image_id="snap-0", image_b64="snap-b64", label="Snapshot 0")],
        frontiers=[AEQAFrontier(frontier_id="4", image_b64="frontier-b64", label="Frontier 4")],
        egocentric_views=[
            AEQAImage(
                image_id="current_view",
                image_b64="current-b64",
                label="Current egocentric view",
                source="egocentric",
            )
        ],
    )

    text_blocks = [
        block.get("text", "")
        for block in build_explore_messages(state)[1]["content"]
        if block.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)

    assert (
        joined.index("Previously Observed Clues")
        < joined.index("Current / Egocentric Views")
        < joined.index("Available Snapshots")
        < joined.index("Available Exploration Directions")
    )


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


class ScriptedVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def call_vlm(self, messages, **kwargs):
        self.messages.append(messages)
        return self.responses.pop(0)


class RecordingAudit:
    def __init__(self):
        self.calls = []

    def record_multimodal(self, **kwargs):
        self.calls.append(kwargs)


def _episode():
    return EpisodeState(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
    )


def test_controller_submits_when_answerer_finds_answer():
    controller = AEQAPredictiveController()
    planner = ScriptedVLM(["Answer: a white towel (Evidence: Snapshot 0)"])
    audit = RecordingAudit()
    env = FakeVisualEnv()

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=env,
        planner=planner,
        prompt_audit=audit,
    )

    assert decision.action_type == "submit_answer"
    assert decision.arguments["answer"] == "a white towel"
    assert decision.arguments["evidence_snapshot"] == 0
    assert len(planner.messages) == 1
    assert audit.calls[0]["label"] == "aeqa_answerer"


def test_controller_explores_frontier_when_answerer_says_continue():
    controller = AEQAPredictiveController()
    planner = ScriptedVLM([
        "Continue Exploration",
        "The kitchen is likely beyond this opening.\nNext Step: Frontier 4",
    ])
    env = FakeVisualEnv()

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=env,
        planner=planner,
    )

    assert decision.action_type == "explore_frontier"
    assert decision.arguments["frontier_id"] == "4"
    assert len(planner.messages) == 2


def test_controller_submits_unanswerable_when_no_frontier_and_no_answer():
    class NoFrontierEnv:
        def get_aeqa_visual_state(self, episode):
            return {
                "question": episode.prompt,
                "current_step": 0,
                "snapshots": [],
                "frontiers": [],
                "egocentric_views": [],
                "memory_text": "",
                "tool_feedback": "",
            }

    controller = AEQAPredictiveController()
    planner = ScriptedVLM(["Continue Exploration"])

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=NoFrontierEnv(),
        planner=planner,
    )

    assert decision.action_type == "submit_answer"
    assert decision.arguments["answer"] == "unanswerable"
    assert decision.confidence == 0.0


def test_controller_memory_cache_is_bounded():
    controller = AEQAPredictiveController(max_memory_episodes=2)
    ep1_memory = controller.memory_for("ep-1")
    ep1_memory.step_summaries.append("old summary")

    controller.memory_for("ep-2")
    controller.memory_for("ep-3")

    assert list(controller._memory_by_episode) == ["ep-2", "ep-3"]
    assert controller.memory_for("ep-1") is not ep1_memory


def test_controller_ignores_frontiers_without_image_payloads():
    class EmptyImageFrontierEnv:
        def get_aeqa_visual_state(self, episode):
            return {
                "question": episode.prompt,
                "current_step": 0,
                "snapshots": [],
                "frontiers": [
                    {"frontier_id": "ghost", "image_b64": "", "label": "Ghost Frontier"},
                ],
                "egocentric_views": [],
                "memory_text": "",
                "tool_feedback": "",
            }

    controller = AEQAPredictiveController()
    planner = ScriptedVLM(["Continue Exploration"])

    decision = controller.decide(
        episode=_episode(),
        context_text="compiled text",
        env=EmptyImageFrontierEnv(),
        planner=planner,
    )

    assert decision.action_type == "submit_answer"
    assert decision.arguments["answer"] == "unanswerable"
    assert len(planner.messages) == 1
