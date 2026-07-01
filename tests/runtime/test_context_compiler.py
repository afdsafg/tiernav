"""Tests for the sectioned context compiler."""
from __future__ import annotations

import hashlib

import pytest

from src.tiernav_runtime.context import ContextCompiler, render_prompt
from src.tiernav_runtime.contracts import (
    ContextSection,
    EpisodeState,
    MemoryPack,
    Observation,
)


def _base_state(**overrides) -> EpisodeState:
    kwargs = dict(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="Where is the lamp?",
        round_index=1,
        step_index=3,
    )
    kwargs.update(overrides)
    return EpisodeState(**kwargs)


def _state_with_memory() -> EpisodeState:
    return _base_state(
        last_observation=Observation(summary="A red chair is visible near the window."),
        memory_pack=MemoryPack(
            query="lamp",
            summary="Lamp previously seen on the desk.",
            evidence_ids=["ev-001", "ev-002"],
            reuse_hint="Reuse the desk landmark.",
        ),
    )


# A string action_schema, matching the plan's `action_schema: str` contract.
SCHEMA = "submit_answer, explore_frontier"


def test_sections_ordered_cacheable_first():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    names = [s.name for s in sections]
    expected_order = [
        "task_instruction",
        "action_schema",
        "memory_index",
        "task_state",
        "recent_trace",
        "current_observation",
        "scene_graph_memory",
        "available_targets",
        "tool_feedback",
        "policy_hint",
    ]
    assert names == expected_order

    # First three are cacheable, at least one dynamic section after.
    assert all(s.cacheable for s in sections[:3])
    assert any(not s.cacheable for s in sections)


def test_task_state_is_only_cache_break_section():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    cache_break_names = [s.name for s in sections if s.cache_break]
    assert cache_break_names == ["task_state"]
    # All cacheable sections keep cache_break=False.
    assert all(not s.cache_break for s in sections if s.cacheable)


def test_cache_break_field_defaults_false_on_model():
    section = ContextSection(name="x", content="y", cacheable=True)
    assert section.cache_break is False


def test_required_section_names_present():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    names = {s.name for s in sections}
    assert {
        "task_instruction",
        "action_schema",
        "memory_index",
        "task_state",
        "recent_trace",
        "current_observation",
        "scene_graph_memory",
        "tool_feedback",
        "policy_hint",
    } <= names


def test_action_schema_section_content_is_raw_string():
    """action_schema is `str`; section.content must equal the input verbatim,
    not a JSON-quoted string."""
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema="schema")

    action = {s.name: s for s in sections}["action_schema"]
    assert action.content == "schema"
    # Guard against the old json.dumps behavior regressing.
    assert action.content != '"schema"'


def test_task_instruction_does_not_contain_fake_target_ids():
    """Static examples must not provide IDs that are absent from available_targets."""
    state = _base_state()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    task = {s.name: s for s in sections}["task_instruction"]

    assert '"frontier_id": "0"' not in task.content
    assert '"seed_id": "0"' not in task.content
    assert '"object_name": "chair"' not in task.content


def test_task_instruction_forbids_target_tools_without_listed_targets():
    state = _base_state()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    task = {s.name: s for s in sections}["task_instruction"]

    assert "Do not call explore_frontier when frontiers is none or absent." in task.content
    assert "Do not call explore_seed when seeds is none or absent." in task.content
    assert "Do not call navigate_to_object when objects is none or absent." in task.content


def test_same_input_gives_stable_content_hash():
    state = _state_with_memory()
    compiler = ContextCompiler()

    a = compiler.compile(state, action_schema=SCHEMA)
    b = compiler.compile(state, action_schema=SCHEMA)

    assert [s.content_hash for s in a] == [s.content_hash for s in b]
    # Hashes are non-empty sha256 hex strings for non-empty content.
    for s in a:
        if s.content:
            assert s.content_hash == hashlib.sha256(s.content.encode("utf-8")).hexdigest()
            assert len(s.content_hash) == 64


def test_content_hash_changes_when_section_content_changes_but_unchanged_preserved():
    state = _state_with_memory()
    compiler = ContextCompiler()

    schema_v1 = "submit_answer, explore_frontier"
    schema_v2 = "submit_answer, explore_frontier, look_around"
    a = compiler.compile(state, action_schema=schema_v1)
    b = compiler.compile(state, action_schema=schema_v2)

    by_name_a = {s.name: s for s in a}
    by_name_b = {s.name: s for s in b}

    # action_schema content differs -> hash changes.
    assert by_name_a["action_schema"].content != by_name_b["action_schema"].content
    assert by_name_a["action_schema"].content_hash != by_name_b["action_schema"].content_hash

    # task_instruction, memory_index, recent_trace unchanged -> same hashes.
    for name in ("task_instruction", "memory_index", "recent_trace", "current_observation"):
        assert by_name_a[name].content_hash == by_name_b[name].content_hash, name


def test_include_memory_false_makes_memory_index_content_empty():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections_with = compiler.compile(state, action_schema=SCHEMA, include_memory=True)
    sections_without = compiler.compile(state, action_schema=SCHEMA, include_memory=False)

    mem_with = {s.name: s for s in sections_with}["memory_index"]
    mem_without = {s.name: s for s in sections_without}["memory_index"]

    assert mem_with.content != ""
    assert "Lamp previously seen on the desk." in mem_with.content
    assert "ev-001" in mem_with.content
    assert "Reuse the desk landmark." in mem_with.content

    assert mem_without.content == ""


def test_memory_index_empty_when_no_memory_pack():
    state = _base_state()  # no memory_pack
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA, include_memory=True)

    mem = {s.name: s for s in sections}["memory_index"]
    assert mem.content == ""


def test_current_observation_contains_last_observation_summary():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    obs = {s.name: s for s in sections}["current_observation"]
    assert "A red chair is visible near the window." in obs.content
    assert not obs.cacheable


def test_recent_trace_contains_round_and_step_index():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    trace = {s.name: s for s in sections}["recent_trace"]
    assert "1" in trace.content  # round_index
    assert "3" in trace.content  # step_index
    assert not trace.cacheable


def test_policy_hint_is_dynamic_and_rendered_when_non_empty():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections_empty = compiler.compile(state, action_schema=SCHEMA, policy_hint="")
    sections_set = compiler.compile(
        state, action_schema=SCHEMA, policy_hint="Prefer explore actions."
    )

    hint_empty = {s.name: s for s in sections_empty}["policy_hint"]
    hint_set = {s.name: s for s in sections_set}["policy_hint"]

    assert not hint_set.cacheable
    assert hint_empty.content == ""
    assert "Prefer explore actions." in hint_set.content

    rendered_empty = render_prompt(sections_empty)
    rendered_set = render_prompt(sections_set)
    assert "Prefer explore actions." not in rendered_empty
    assert "Prefer explore actions." in rendered_set


def test_render_prompt_includes_memory_pack_and_observation():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    rendered = render_prompt(sections)

    assert "Lamp previously seen on the desk." in rendered
    assert "ev-001" in rendered
    assert "Reuse the desk landmark." in rendered
    assert "A red chair is visible near the window." in rendered
    # action schema content present too.
    assert "explore_frontier" in rendered


def test_tool_feedback_warns_after_repeated_panorama_without_movement():
    """Panorama is useful once; repeated in-place observations need a move hint."""
    state = _base_state(
        round_index=2,
        step_index=2,
        current_decision={
            "action_type": "explore_panorama",
            "arguments": {},
        },
        last_observation=Observation(
            summary="Panorama: 8 views at step 2",
            object_ids=["chair"],
            raw={
                "outcome": "panorama_complete",
                "progress": "Panorama: 8 views at step 2",
                "path_length": 0.0,
            },
        ),
    )
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    feedback = {s.name: s for s in sections}["tool_feedback"]

    assert "last_tool_action: explore_panorama" in feedback.content
    assert "choose a valid available target and move" in feedback.content
    assert "objects_nearby: chair" in feedback.content


def test_tool_feedback_tolerates_non_numeric_path_length():
    state = _base_state(
        round_index=2,
        step_index=2,
        current_decision={"action_type": "explore_panorama"},
        last_observation=Observation(
            summary="Panorama complete",
            raw={
                "outcome": "panorama_complete",
                "path_length": "unknown",
            },
        ),
    )
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    feedback = {s.name: s for s in sections}["tool_feedback"]

    assert "last_tool_action: explore_panorama" in feedback.content


def test_tool_feedback_uses_path_delta_for_stationary_panorama():
    state = _base_state(
        round_index=4,
        step_index=4,
        current_decision={"action_type": "explore_panorama"},
        last_observation=Observation(
            summary="Panorama complete after earlier movement",
            raw={
                "outcome": "panorama_complete",
                "path_length": 3.5,
                "path_delta": 0.0,
            },
        ),
    )
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    feedback = {s.name: s for s in sections}["tool_feedback"]

    assert "choose a valid available target and move" in feedback.content


def test_instance_render_prompt_matches_module_render_prompt():
    """compiler.render_prompt (instance method) must equal module-level render_prompt."""
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    assert compiler.render_prompt(sections) == render_prompt(sections)


def test_instance_render_prompt_chained_with_compile():
    """Plan-style call: compiler.render_prompt(compiler.compile(state, action_schema=...)).

    Must not raise AttributeError and must include memory + observation content.
    """
    state = _state_with_memory()
    compiler = ContextCompiler()

    prompt = compiler.render_prompt(compiler.compile(state, action_schema="schema"))

    assert isinstance(prompt, str)
    # Memory pack content rendered.
    assert "Lamp previously seen on the desk." in prompt
    # Observation content rendered.
    assert "A red chair is visible near the window." in prompt
    # action_schema string rendered verbatim, not JSON-quoted.
    assert "schema" in prompt
    assert '"schema"' not in prompt


def test_render_prompt_skips_empty_sections():
    state = _base_state()  # no memory_pack, empty observation summary
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    rendered = render_prompt(sections)

    # Empty policy_hint and empty memory_index must not contribute a header with
    # no body — render_prompt only renders non-empty content.
    assert rendered.strip() != ""  # task_instruction / action_schema / recent_trace remain
    # No orphan headers for empty memory section.
    lines = [ln for ln in rendered.splitlines() if ln.strip()]
    for ln in lines:
        assert ln.strip() != ""


def test_token_estimate_zero_for_empty_content_positive_otherwise():
    state = _base_state()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    for s in sections:
        if s.content == "":
            assert s.token_estimate == 0, s.name
        else:
            assert s.token_estimate > 0, s.name


def test_returns_context_section_instances():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    assert all(isinstance(s, ContextSection) for s in sections)


def test_content_hash_empty_string_for_empty_content():
    state = _base_state()  # empty memory, empty observation summary
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)

    for s in sections:
        if s.content == "":
            assert s.content_hash == ""


def test_compile_rejects_non_str_action_schema():
    """Non-str action_schema must raise TypeError at the compile entry point,
    not leak as an AttributeError from _estimate_tokens internals."""
    state = _state_with_memory()
    compiler = ContextCompiler()

    with pytest.raises(TypeError) as excinfo:
        compiler.compile(state, action_schema=123)

    msg = str(excinfo.value)
    assert "action_schema" in msg
    assert "str" in msg


def test_compile_rejects_non_str_policy_hint():
    """Non-str policy_hint must raise TypeError at the compile entry point,
    not leak as an AttributeError from _estimate_tokens internals."""
    state = _state_with_memory()
    compiler = ContextCompiler()

    with pytest.raises(TypeError) as excinfo:
        compiler.compile(state, action_schema=SCHEMA, policy_hint=123)

    msg = str(excinfo.value)
    assert "policy_hint" in msg
    assert "str" in msg


# --- Scoring-only field exclusion ------------------------------------------


def test_scoring_only_goal_object_ids_never_leak_into_prompt():
    """GOATBench goal_object_ids_for_scoring are scorer-authoritative and must
    NOT appear anywhere in the planner-visible prompt.

    The compiler reads only from EpisodeState (prompt + identity fields), and
    EpisodeState does not carry goal_metadata. So even if the originating
    EpisodeRequest carried scoring ids in goal_metadata, they cannot reach the
    prompt. This test pins that contract: the rendered prompt must contain the
    planner-visible goal description but must NOT contain the scoring ids.
    """
    scoring_ids = ["obj-target-42", "obj-target-99"]
    planner_visible_description = "Find the red mug on the kitchen counter."

    # EpisodeState.prompt carries only the planner-visible description.
    state = _base_state(prompt=planner_visible_description)
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    rendered = render_prompt(sections)

    assert planner_visible_description in rendered
    for scoring_id in scoring_ids:
        assert scoring_id not in rendered, (
            f"scoring-only id {scoring_id!r} leaked into planner prompt"
        )


def test_task_instruction_renders_only_planner_safe_fields():
    """task_instruction section must render only: episode_id, scene_id,
    task_name, task_mode, prompt — never goal_metadata / scoring ids."""
    state = _base_state(prompt="Where is the lamp?")
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    task = {s.name: s for s in sections}["task_instruction"]

    # Planner-safe identity fields are present.
    assert "episode_id: ep-1" in task.content
    assert "scene_id: scene-1" in task.content
    assert "task_name: aeqa" in task.content
    assert "task_mode: question_answering" in task.content
    assert "prompt: Where is the lamp?" in task.content
    # No goal_metadata / scoring surface is rendered.
    assert "goal_metadata" not in task.content
    assert "goal_object_ids_for_scoring" not in task.content


def test_task_state_contains_agent_workflow_guidance():
    state = _base_state(task_name="goatbench", task_mode="goal_navigation")
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    task_state = {s.name: s for s in sections}["task_state"]

    assert "continuous_context" in task_state.content
    assert "goal_navigation" in task_state.content
    assert "Do not repeat failed actions" in task_state.content
    assert not task_state.cacheable


class _SceneGraph:
    def get_manifest(self):
        return "rooms: 2(partially_explored)\nobjects: sofa x1\nevidence: 1 entries"


class _EnvWithSceneGraph:
    scene_graph_memory = _SceneGraph()


def test_scene_graph_memory_section_uses_env_scene_graph_manifest():
    state = _base_state()
    compiler = ContextCompiler()

    sections = compiler.compile(
        state,
        action_schema=SCHEMA,
        env=_EnvWithSceneGraph(),
    )
    scene_graph = {s.name: s for s in sections}["scene_graph_memory"]

    assert "2(partially_explored)" in scene_graph.content
    assert "sofa" in scene_graph.content
    assert not scene_graph.cacheable


def test_scene_graph_memory_section_appends_recalled_memory():
    state = _base_state(recalled_memory="- room 2 (goal): {\"notes\": \"kitchen\"}")
    compiler = ContextCompiler()

    sections = compiler.compile(
        state,
        action_schema=SCHEMA,
        env=_EnvWithSceneGraph(),
    )
    scene_graph = {s.name: s for s in sections}["scene_graph_memory"]

    assert "recalled_details:" in scene_graph.content
    assert "- room 2 (goal)" in scene_graph.content


def test_tool_feedback_section_surfaces_failed_tool_result():
    state = _base_state(
        current_decision=None,
        last_observation=Observation(
            summary="Navigation status: no path to frontier 7",
            raw={
                "outcome": "target_not_reached",
                "subgoal": "Navigate to frontier 7",
                "progress": "No valid path",
            },
        ),
    )
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    feedback = {s.name: s for s in sections}["tool_feedback"]

    assert "target_not_reached" in feedback.content
    assert "Navigate to frontier 7" in feedback.content
    assert "try a different target" in feedback.content


def test_available_targets_does_not_print_diagnostics(capsys):
    state = _base_state()
    compiler = ContextCompiler()

    compiler.compile(state, action_schema=SCHEMA, env=object())

    captured = capsys.readouterr()
    assert "[DIAG" not in captured.err


class _RoomRegion:
    def __init__(self, room_id: int):
        self.room_id = room_id


class _TsdfWithRooms:
    frontiers = []
    seeds = []
    room_regions = [_RoomRegion(2), _RoomRegion(5)]


class _EnvWithRoomRegions:
    tsdf_planner = _TsdfWithRooms()


def test_available_targets_uses_room_regions_as_seed_targets():
    state = _base_state(task_name="goatbench", task_mode="goal_navigation")
    compiler = ContextCompiler()

    sections = compiler.compile(
        state,
        action_schema=SCHEMA,
        env=_EnvWithRoomRegions(),
    )
    targets = {s.name: s for s in sections}["available_targets"]

    assert "seeds: 2, 5" in targets.content


def test_available_targets_marks_missing_frontiers_as_none_when_seeds_exist():
    state = _base_state(task_name="goatbench", task_mode="goal_navigation")
    compiler = ContextCompiler()

    sections = compiler.compile(
        state,
        action_schema=SCHEMA,
        env=_EnvWithRoomRegions(),
    )
    targets = {s.name: s for s in sections}["available_targets"]

    assert "frontiers: none" in targets.content


# --- Phase 2: compact_summary in task_state --------------------------------


def test_task_state_includes_compact_summary_when_set():
    state = _base_state(compact_summary="Rooms visited: kitchen. Mug found on counter.")
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    task_state = {s.name: s for s in sections}["task_state"]

    assert "compact_summary:" in task_state.content
    assert "Rooms visited: kitchen. Mug found on counter." in task_state.content


def test_task_state_omits_compact_summary_when_empty():
    state = _base_state(compact_summary="")
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema=SCHEMA)
    task_state = {s.name: s for s in sections}["task_state"]

    assert "compact_summary:" not in task_state.content

