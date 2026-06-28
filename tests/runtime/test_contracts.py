"""Contract tests for the TierNav runtime."""
import json

from pydantic import ValidationError

from src.tiernav_runtime.contracts import (
    AblationConfig,
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    Observation,
    PlannerDecision,
    RunSpec,
    ToolCall,
    ToolResult,
    dump_runtime_json_schemas,
)


def test_run_spec_has_research_ablation_axes():
    spec = RunSpec(
        run_id="run-001",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
        seed=7,
        ablation=AblationConfig(
            continuous_context=True,
            spatial_memory=True,
            active_memory_query=True,
            prompt_cache=True,
            stall_recovery=False,
        ),
    )

    assert spec.ablation.continuous_context is True
    assert spec.ablation.spatial_memory is True
    assert spec.ablation.active_memory_query is True


def test_run_spec_rejects_unknown_schema_version():
    try:
        RunSpec(
            schema_version="future.v99",
            run_id="run-001",
            task_name="aeqa",
            dataset_split="dev",
            output_dir="/tmp/tiernav",
            planner_provider="mimo",
            planner_model="qwen3-vl-flash",
        )
    except ValidationError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("RunSpec accepted an unknown schema_version")


def test_episode_request_rejects_unknown_task_mode():
    try:
        EpisodeRequest(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="unknown",
            prompt="What color is the chair?",
        )
    except ValidationError as exc:
        assert "task_mode" in str(exc)
    else:
        raise AssertionError("EpisodeRequest accepted an unknown task_mode")


def test_planner_decision_round_trip_json():
    decision = PlannerDecision(
        action_type="navigate_to_object",
        reasoning="The chair is visible.",
        expected="Move closer to verify the answer.",
        confidence=0.8,
        arguments={"snapshot_id": "step1_view0", "object_name": "chair"},
    )

    encoded = decision.model_dump_json()
    decoded = PlannerDecision.model_validate_json(encoded)

    assert decoded.action_type == "navigate_to_object"
    assert decoded.arguments["object_name"] == "chair"


def test_planner_decision_clamps_out_of_range_confidence():
    decision_low = PlannerDecision(action_type="search", confidence=-0.2)
    decision_high = PlannerDecision(action_type="search", confidence=1.7)

    assert decision_low.confidence == 0.0
    assert decision_high.confidence == 1.0


def test_episode_state_serializes_without_numpy_objects():
    state = EpisodeState(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="Where is the lamp?",
        round_index=1,
        step_index=2,
        pose={"x": 1.0, "y": 0.0, "z": 2.0, "yaw": 0.5},
    )

    payload = json.loads(state.model_dump_json())

    assert payload["pose"]["x"] == 1.0
    assert payload["round_index"] == 1


def test_episode_state_rejects_unknown_top_level_fields():
    try:
        EpisodeState(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="Where is the lamp?",
            unexpected=True,
        )
    except ValidationError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("EpisodeState accepted an unexpected top-level field")


def test_tool_contracts_validate_terminal_results():
    call = ToolCall(
        call_id="tool-1",
        action_type="submit_answer",
        arguments={"answer": "red"},
    )
    result = ToolResult(
        call_id=call.call_id,
        action_type=call.action_type,
        ok=True,
        terminal=True,
        observation=Observation(summary="Answer submitted."),
    )

    assert result.terminal is True
    assert result.observation.summary == "Answer submitted."


def test_tool_result_rejects_unknown_nested_observation_fields():
    try:
        ToolResult(
            call_id="tool-1",
            action_type="submit_answer",
            ok=True,
            observation={"summary": "Answer submitted.", "unexpected": "bad"},
        )
    except ValidationError as exc:
        message = str(exc)
        assert "observation" in message
        assert "unexpected" in message
    else:
        raise AssertionError("ToolResult accepted an unexpected nested observation field")


def test_episode_result_has_common_metrics_for_aeqa_and_goatbench():
    result = EpisodeResult(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        success=True,
        answer="chair",
        steps_taken=4,
        rounds_used=2,
        path_length=3.5,
        event_log_path="/tmp/tiernav/ep-1/events.jsonl",
    )

    assert result.path_length == 3.5
    assert result.event_log_path.endswith("events.jsonl")


def test_observation_rejects_unknown_nested_fields():
    try:
        Observation(summary="Seen chair", unknown_nested_field="x")
    except ValidationError as exc:
        assert "unknown_nested_field" in str(exc)
    else:
        raise AssertionError("Observation accepted an unexpected nested field")


def test_json_schema_dump_contains_all_public_models():
    schemas = dump_runtime_json_schemas()

    assert set(schemas) == {
        "RunSpec",
        "EpisodeRequest",
        "EpisodeState",
        "EpisodeResult",
        "PlannerDecision",
        "ToolCall",
        "ToolResult",
        "Observation",
        "MemoryPack",
        "ContextSection",
    }
    assert schemas["RunSpec"]["type"] == "object"
    assert schemas["PlannerDecision"]["properties"]["confidence"]["minimum"] == 0.0
    assert schemas["PlannerDecision"]["properties"]["confidence"]["maximum"] == 1.0
