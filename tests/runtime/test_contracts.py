"""Contract tests for the TierNav runtime."""
import json
from typing import get_args

import pytest
from pydantic import ValidationError

from src.tiernav_runtime.contracts import (
    AblationConfig,
    BenchmarkRule,
    ContextSection,
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    GoalSpec,
    MemoryPack,
    MemoryScope,
    Observation,
    PlannerDecision,
    PUBLIC_MODELS,
    PublicModel,
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


def test_run_spec_rejects_negative_max_rounds():
    try:
        RunSpec(
            run_id="run-001",
            task_name="aeqa",
            dataset_split="dev",
            output_dir="/tmp/tiernav",
            planner_provider="mimo",
            planner_model="qwen3-vl-flash",
            max_rounds=-1,
        )
    except ValidationError as exc:
        assert "max_rounds" in str(exc)
    else:
        raise AssertionError("RunSpec accepted a negative max_rounds")


def test_run_spec_rejects_string_max_rounds():
    try:
        RunSpec(
            run_id="run-001",
            task_name="aeqa",
            dataset_split="dev",
            output_dir="/tmp/tiernav",
            planner_provider="mimo",
            planner_model="qwen3-vl-flash",
            max_rounds="5",
        )
    except ValidationError as exc:
        assert "max_rounds" in str(exc)
    else:
        raise AssertionError("RunSpec accepted string max_rounds")


def test_run_spec_rejects_non_json_metadata_values():
    try:
        RunSpec(
            run_id="run-001",
            task_name="aeqa",
            dataset_split="dev",
            output_dir="/tmp/tiernav",
            planner_provider="mimo",
            planner_model="qwen3-vl-flash",
            metadata={"obj": object()},
        )
    except ValidationError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("RunSpec accepted a non-JSON metadata value")


def test_run_spec_rejects_non_finite_json_metadata_values():
    try:
        RunSpec(
            run_id="run-001",
            task_name="aeqa",
            dataset_split="dev",
            output_dir="/tmp/tiernav",
            planner_provider="mimo",
            planner_model="qwen3-vl-flash",
            metadata={"bad": float("nan")},
        )
    except ValidationError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("RunSpec accepted a non-finite JSON metadata value")


@pytest.mark.parametrize(
    ("model_type", "kwargs"),
    [
        (
            RunSpec,
            {
                "run_id": "run-001",
                "task_name": "aeqa",
                "dataset_split": "dev",
                "output_dir": "/tmp/tiernav",
                "planner_provider": "mimo",
                "planner_model": "qwen3-vl-flash",
            },
        ),
        (
            EpisodeRequest,
            {
                "episode_id": "ep-1",
                "scene_id": "scene",
                "task_name": "aeqa",
                "task_mode": "question_answering",
                "prompt": "What color is the chair?",
            },
        ),
        (
            EpisodeState,
            {
                "episode_id": "ep-1",
                "scene_id": "scene",
                "task_name": "aeqa",
                "task_mode": "question_answering",
                "prompt": "Where is the lamp?",
            },
        ),
        (
            EpisodeResult,
            {
                "episode_id": "ep-1",
                "scene_id": "scene",
                "task_name": "aeqa",
                "task_mode": "question_answering",
                "success": True,
            },
        ),
    ],
)
def test_versioned_models_reject_unknown_schema_version(model_type, kwargs):
    try:
        model_type(
            schema_version="future.v99",
            **kwargs,
        )
    except ValidationError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError(f"{model_type.__name__} accepted an unknown schema_version")


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


def test_episode_request_rejects_non_finite_initial_pose_values():
    try:
        EpisodeRequest(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="What color is the chair?",
            initial_pose={"x": float("nan")},
        )
    except ValidationError as exc:
        assert "initial_pose" in str(exc)
    else:
        raise AssertionError("EpisodeRequest accepted a non-finite initial_pose value")


def test_episode_request_rejects_non_json_goal_metadata_values():
    try:
        EpisodeRequest(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="What color is the chair?",
            goal_metadata={"obj": object()},
        )
    except ValidationError as exc:
        assert "goal_metadata" in str(exc)
    else:
        raise AssertionError("EpisodeRequest accepted a non-JSON goal_metadata value")


def test_episode_request_rejects_non_finite_goal_metadata_values():
    try:
        EpisodeRequest(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="What color is the chair?",
            goal_metadata={"bad": float("inf")},
        )
    except ValidationError as exc:
        assert "goal_metadata" in str(exc)
    else:
        raise AssertionError("EpisodeRequest accepted a non-finite goal_metadata value")


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


def test_planner_decision_rejects_non_json_arguments_values():
    try:
        PlannerDecision(action_type="search", arguments={"obj": object()})
    except ValidationError as exc:
        assert "arguments" in str(exc)
    else:
        raise AssertionError("PlannerDecision accepted a non-JSON arguments value")


def test_planner_decision_rejects_non_finite_json_arguments_values():
    try:
        PlannerDecision(action_type="search", arguments={"bad": float("inf")})
    except ValidationError as exc:
        assert "arguments" in str(exc)
    else:
        raise AssertionError("PlannerDecision accepted a non-finite JSON arguments value")


def test_planner_decision_serializes_nested_json_arguments():
    decision = PlannerDecision(
        action_type="search",
        arguments={
            "target": {
                "name": "chair",
                "attributes": ["red", 2, True, None],
            }
        },
    )

    payload = json.loads(decision.model_dump_json())

    assert payload["arguments"]["target"]["name"] == "chair"
    assert payload["arguments"]["target"]["attributes"] == ["red", 2, True, None]


def test_json_payload_fields_serialize_finite_numbers_as_json_numbers():
    spec = RunSpec(
        run_id="run-001",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
        metadata={"count": 2, "score": 1.5, "nested": {"values": [3, 4.25]}},
    )

    payload = json.loads(spec.model_dump_json())

    assert payload["metadata"]["count"] == 2
    assert payload["metadata"]["score"] == 1.5
    assert payload["metadata"]["nested"]["values"] == [3, 4.25]
    assert payload["metadata"]["score"] is not None


def test_planner_decision_clamps_out_of_range_confidence():
    decision_low = PlannerDecision(action_type="search", confidence=-0.2)
    decision_high = PlannerDecision(action_type="search", confidence=1.7)

    assert decision_low.confidence == 0.0
    assert decision_high.confidence == 1.0


def test_planner_decision_rejects_string_confidence():
    try:
        PlannerDecision(action_type="search", confidence="0.6")
    except ValidationError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("PlannerDecision accepted string confidence")


def test_planner_decision_rejects_bool_confidence():
    try:
        PlannerDecision(action_type="search", confidence=True)
    except ValidationError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("PlannerDecision accepted bool confidence")


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


def test_episode_state_rejects_negative_step_index():
    try:
        EpisodeState(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="Where is the lamp?",
            step_index=-1,
        )
    except ValidationError as exc:
        assert "step_index" in str(exc)
    else:
        raise AssertionError("EpisodeState accepted a negative step_index")


def test_episode_state_rejects_bool_step_index():
    try:
        EpisodeState(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="Where is the lamp?",
            step_index=True,
        )
    except ValidationError as exc:
        assert "step_index" in str(exc)
    else:
        raise AssertionError("EpisodeState accepted bool step_index")


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


def test_episode_state_rejects_unknown_nested_current_decision_fields():
    try:
        EpisodeState(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="Where is the lamp?",
            current_decision={"action_type": "look", "unexpected": "bad"},
        )
    except ValidationError as exc:
        message = str(exc)
        assert "current_decision" in message
        assert "unexpected" in message
    else:
        raise AssertionError("EpisodeState accepted an unexpected current_decision field")


def test_episode_state_rejects_non_finite_pose_values():
    try:
        EpisodeState(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="Where is the lamp?",
            pose={"x": float("inf")},
        )
    except ValidationError as exc:
        assert "pose" in str(exc)
    else:
        raise AssertionError("EpisodeState accepted a non-finite pose value")


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


def test_tool_call_rejects_non_json_arguments_values():
    try:
        ToolCall(
            call_id="tool-1",
            action_type="submit_answer",
            arguments={"obj": object()},
        )
    except ValidationError as exc:
        assert "arguments" in str(exc)
    else:
        raise AssertionError("ToolCall accepted a non-JSON arguments value")


def test_tool_result_rejects_non_finite_metric_values():
    try:
        ToolResult(
            call_id="tool-1",
            action_type="submit_answer",
            ok=True,
            metrics={"distance": float("nan")},
        )
    except ValidationError as exc:
        assert "metrics" in str(exc)
    else:
        raise AssertionError("ToolResult accepted a non-finite metric value")


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


def test_episode_result_rejects_negative_steps_taken():
    try:
        EpisodeResult(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            success=True,
            steps_taken=-1,
        )
    except ValidationError as exc:
        assert "steps_taken" in str(exc)
    else:
        raise AssertionError("EpisodeResult accepted negative steps_taken")


def test_episode_result_rejects_negative_path_length():
    try:
        EpisodeResult(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            success=True,
            path_length=-0.5,
        )
    except ValidationError as exc:
        assert "path_length" in str(exc)
    else:
        raise AssertionError("EpisodeResult accepted negative path_length")


@pytest.mark.parametrize("invalid_path_length", [float("inf"), float("nan")])
def test_episode_result_rejects_non_finite_path_length(invalid_path_length):
    try:
        EpisodeResult(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="question_answering",
            success=True,
            path_length=invalid_path_length,
        )
    except ValidationError as exc:
        assert "path_length" in str(exc)
    else:
        raise AssertionError("EpisodeResult accepted a non-finite path_length")


def test_episode_result_serializes_finite_path_length_as_json_number():
    result = EpisodeResult(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        success=True,
        path_length=3.5,
    )

    payload = json.loads(result.model_dump_json())

    assert payload["path_length"] == 3.5
    assert payload["path_length"] is not None


def test_observation_rejects_unknown_nested_fields():
    try:
        Observation(summary="Seen chair", unknown_nested_field="x")
    except ValidationError as exc:
        assert "unknown_nested_field" in str(exc)
    else:
        raise AssertionError("Observation accepted an unexpected nested field")


def test_observation_rejects_non_json_raw_values():
    try:
        Observation(raw={"obj": object()})
    except ValidationError as exc:
        assert "raw" in str(exc)
    else:
        raise AssertionError("Observation accepted a non-JSON raw value")


def test_observation_rejects_non_finite_json_raw_values():
    try:
        Observation(raw={"bad": float("nan")})
    except ValidationError as exc:
        assert "raw" in str(exc)
    else:
        raise AssertionError("Observation accepted a non-finite JSON raw value")


def test_observation_rejects_non_finite_pose_values():
    try:
        Observation(summary="Seen chair", pose={"x": float("inf")})
    except ValidationError as exc:
        assert "pose" in str(exc)
    else:
        raise AssertionError("Observation accepted a non-finite pose value")


@pytest.mark.parametrize("invalid_confidence", [float("nan"), float("inf")])
def test_memory_pack_rejects_non_finite_confidence(invalid_confidence):
    try:
        MemoryPack(query="lamp", summary="Seen lamp", confidence=invalid_confidence)
    except ValidationError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("MemoryPack accepted a non-finite confidence")


@pytest.mark.parametrize("invalid_confidence", ["0.6", True])
def test_memory_pack_rejects_nonnumeric_confidence(invalid_confidence):
    try:
        MemoryPack(query="lamp", summary="Seen lamp", confidence=invalid_confidence)
    except ValidationError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("MemoryPack accepted a nonnumeric confidence")


def test_memory_pack_serializes_numeric_confidence_as_json_number():
    pack = MemoryPack(query="lamp", summary="Seen lamp", confidence=0.6)

    payload = json.loads(pack.model_dump_json())

    assert payload["confidence"] == 0.6
    assert payload["confidence"] is not None


def test_mapped_float_values_serialize_as_json_numbers():
    request = EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="Where is the lamp?",
        initial_pose={"x": 1.25},
    )
    observation = Observation(summary="Seen chair", pose={"x": 2.5})
    tool_result = ToolResult(
        call_id="tool-1",
        action_type="submit_answer",
        ok=True,
        metrics={"distance": 3.75},
    )

    request_payload = json.loads(request.model_dump_json())
    observation_payload = json.loads(observation.model_dump_json())
    tool_result_payload = json.loads(tool_result.model_dump_json())

    assert request_payload["initial_pose"]["x"] == 1.25
    assert request_payload["initial_pose"]["x"] is not None
    assert observation_payload["pose"]["x"] == 2.5
    assert observation_payload["pose"]["x"] is not None
    assert tool_result_payload["metrics"]["distance"] == 3.75
    assert tool_result_payload["metrics"]["distance"] is not None


def test_tool_call_rejects_non_finite_json_arguments_values():
    try:
        ToolCall(call_id="tool-1", action_type="search", arguments={"bad": float("nan")})
    except ValidationError as exc:
        assert "arguments" in str(exc)
    else:
        raise AssertionError("ToolCall accepted a non-finite JSON arguments value")


def test_context_section_rejects_negative_token_estimate():
    try:
        ContextSection(name="planner", content="...", cacheable=True, token_estimate=-1)
    except ValidationError as exc:
        assert "token_estimate" in str(exc)
    else:
        raise AssertionError("ContextSection accepted a negative token_estimate")


def test_context_section_rejects_string_token_estimate():
    try:
        ContextSection(name="planner", content="...", cacheable=True, token_estimate="12")
    except ValidationError as exc:
        assert "token_estimate" in str(exc)
    else:
        raise AssertionError("ContextSection accepted string token_estimate")


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
        "GoalSpec",
        "BenchmarkRule",
    }
    assert schemas["RunSpec"]["type"] == "object"
    assert schemas["RunSpec"]["properties"]["max_rounds"]["minimum"] == 0
    assert schemas["EpisodeState"]["properties"]["step_index"]["minimum"] == 0
    assert schemas["EpisodeResult"]["properties"]["path_length"]["minimum"] == 0.0
    assert schemas["PlannerDecision"]["properties"]["confidence"]["minimum"] == 0.0
    assert schemas["PlannerDecision"]["properties"]["confidence"]["maximum"] == 1.0
    assert schemas["MemoryPack"]["properties"]["confidence"]["minimum"] == 0.0
    assert schemas["MemoryPack"]["properties"]["confidence"]["maximum"] == 1.0
    assert schemas["ContextSection"]["properties"]["token_estimate"]["minimum"] == 0


def test_public_model_literal_matches_public_models_registry():
    assert set(get_args(PublicModel)) == set(PUBLIC_MODELS)


def test_goal_spec_separates_planner_and_scoring_fields():
    goal = GoalSpec(
        goal_type="object",
        goal_description="chair",
        goal_object_ids_for_scoring=["obj-1"],
        subtask_index=2,
        subtask_total=5,
    )
    assert goal.goal_description == "chair"
    assert goal.goal_object_ids_for_scoring == ["obj-1"]


def test_goal_spec_defaults_optional_scoring_fields():
    goal = GoalSpec(goal_type="object", goal_description="chair")

    assert goal.goal_object_ids_for_scoring == []
    assert goal.subtask_index == 0
    assert goal.subtask_total == 0


def test_goal_spec_rejects_unknown_fields():
    try:
        GoalSpec(goal_type="object", goal_description="chair", unexpected=True)
    except ValidationError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("GoalSpec accepted an unexpected field")


def test_benchmark_rule_exposes_memory_scope_and_success_distance():
    rule = BenchmarkRule(
        success_distance_m=1.0,
        requires_explicit_stop=True,
        memory_scope=MemoryScope.SUBTASK_SEQUENCE,
        scoring_mode="distance",
    )
    assert rule.success_distance_m == 1.0


def test_benchmark_rule_defaults_require_explicit_stop_false():
    rule = BenchmarkRule(
        success_distance_m=1.0,
        memory_scope=MemoryScope.SUBTASK_SEQUENCE,
        scoring_mode="distance",
    )
    assert rule.requires_explicit_stop is False


def test_benchmark_rule_rejects_unknown_memory_scope():
    try:
        BenchmarkRule(
            success_distance_m=1.0,
            memory_scope="galactic",
            scoring_mode="distance",
        )
    except ValidationError as exc:
        assert "memory_scope" in str(exc)
    else:
        raise AssertionError("BenchmarkRule accepted an unknown memory_scope")


def test_memory_scope_has_per_question_and_subtask_sequence_values():
    values = {member.value for member in MemoryScope}
    assert "per_question" in values
    assert "subtask_sequence" in values
