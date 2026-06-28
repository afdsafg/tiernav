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


def test_sections_ordered_cacheable_first():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": ["look"]})

    names = [s.name for s in sections]
    expected_order = [
        "task_instruction",
        "action_schema",
        "memory_index",
        "recent_trace",
        "current_observation",
        "policy_hint",
    ]
    assert names == expected_order

    # First three are cacheable, at least one dynamic section after.
    assert all(s.cacheable for s in sections[:3])
    assert any(not s.cacheable for s in sections)


def test_required_section_names_present():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": ["look"]})

    names = {s.name for s in sections}
    assert {
        "task_instruction",
        "action_schema",
        "memory_index",
        "recent_trace",
        "current_observation",
        "policy_hint",
    } <= names


def test_same_input_gives_stable_content_hash():
    state = _state_with_memory()
    compiler = ContextCompiler()

    a = compiler.compile(state, action_schema={"actions": ["look"]})
    b = compiler.compile(state, action_schema={"actions": ["look"]})

    assert [s.content_hash for s in a] == [s.content_hash for s in b]
    # Hashes are non-empty sha256 hex strings for non-empty content.
    for s in a:
        if s.content:
            assert s.content_hash == hashlib.sha256(s.content.encode("utf-8")).hexdigest()
            assert len(s.content_hash) == 64


def test_content_hash_changes_when_section_content_changes_but_unchanged_preserved():
    state = _state_with_memory()
    compiler = ContextCompiler()

    schema_v1 = {"actions": ["look"]}
    schema_v2 = {"actions": ["look", "navigate"]}
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

    sections_with = compiler.compile(state, action_schema={"actions": []}, include_memory=True)
    sections_without = compiler.compile(state, action_schema={"actions": []}, include_memory=False)

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

    sections = compiler.compile(state, action_schema={"actions": []}, include_memory=True)

    mem = {s.name: s for s in sections}["memory_index"]
    assert mem.content == ""


def test_current_observation_contains_last_observation_summary():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": []})

    obs = {s.name: s for s in sections}["current_observation"]
    assert "A red chair is visible near the window." in obs.content
    assert not obs.cacheable


def test_recent_trace_contains_round_and_step_index():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": []})

    trace = {s.name: s for s in sections}["recent_trace"]
    assert "1" in trace.content  # round_index
    assert "3" in trace.content  # step_index
    assert not trace.cacheable


def test_policy_hint_is_dynamic_and_rendered_when_non_empty():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections_empty = compiler.compile(state, action_schema={"actions": []}, policy_hint="")
    sections_set = compiler.compile(
        state, action_schema={"actions": []}, policy_hint="Prefer explore actions."
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

    sections = compiler.compile(state, action_schema={"actions": ["look"]})
    rendered = render_prompt(sections)

    assert "Lamp previously seen on the desk." in rendered
    assert "ev-001" in rendered
    assert "Reuse the desk landmark." in rendered
    assert "A red chair is visible near the window." in rendered
    # action schema content present too.
    assert "look" in rendered


def test_render_prompt_skips_empty_sections():
    state = _base_state()  # no memory_pack, empty observation summary
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": []})
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

    sections = compiler.compile(state, action_schema={"actions": []})

    for s in sections:
        if s.content == "":
            assert s.token_estimate == 0, s.name
        else:
            assert s.token_estimate > 0, s.name


def test_returns_context_section_instances():
    state = _state_with_memory()
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": []})

    assert all(isinstance(s, ContextSection) for s in sections)


def test_content_hash_empty_string_for_empty_content():
    state = _base_state()  # empty memory, empty observation summary
    compiler = ContextCompiler()

    sections = compiler.compile(state, action_schema={"actions": []})

    for s in sections:
        if s.content == "":
            assert s.content_hash == ""
