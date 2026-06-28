"""Tests for planner adapter and stable tool registry."""
from __future__ import annotations

import pytest

from src.agent_planner import PlannerAction
from src.tiernav_runtime.contracts import (
    PlannerDecision,
    ToolCall,
    ToolResult,
)
from src.tiernav_runtime.planner import planner_action_to_decision
from src.tiernav_runtime.tools import (
    NoopNavigationTool,
    RuntimeTool,
    SubmitAnswerTool,
    ToolRegistry,
    with_stable_defaults,
)


# ── Planner adapter ────────────────────────────────────────────────────────


def test_planner_adapter_preserves_arguments():
    action = PlannerAction(
        action_type="navigate_to_object",
        reason="go to chair",
        confidence=0.8,
        snapshot_id="step12_view1",
        object_name="chair",
        seed_id=None,
        frontier_id=None,
        view_idx=3,
        answer=None,
        expected="see the chair",
    )
    decision = planner_action_to_decision(action)

    assert isinstance(decision, PlannerDecision)
    assert decision.action_type == "navigate_to_object"
    assert decision.reasoning == "go to chair"
    assert decision.expected == "see the chair"
    assert decision.confidence == pytest.approx(0.8)
    # Non-None fields collected.
    assert decision.arguments["snapshot_id"] == "step12_view1"
    assert decision.arguments["object_name"] == "chair"
    assert decision.arguments["view_idx"] == 3
    # None optional fields dropped.
    assert "seed_id" not in decision.arguments
    assert "frontier_id" not in decision.arguments
    assert "answer" not in decision.arguments


def test_planner_adapter_clamps_confidence():
    high = PlannerAction(action_type="explore_panorama", confidence=2.5)
    low = PlannerAction(action_type="explore_panorama", confidence=-1.0)
    assert planner_action_to_decision(high).confidence == 1.0
    assert planner_action_to_decision(low).confidence == 0.0


def test_planner_adapter_defaults_expected_empty():
    action = PlannerAction(action_type="explore_panorama", reason="reorient")
    decision = planner_action_to_decision(action)
    assert decision.expected == ""
    assert decision.reasoning == "reorient"
    assert decision.arguments == {}


# ── ToolRegistry dispatch ──────────────────────────────────────────────────


class EchoTool(RuntimeTool):
    name = "echo"
    terminal = False

    def run(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            action_type=call.action_type,
            ok=True,
            terminal=False,
            observation={"summary": "echo"},  # type: ignore[arg-type]
        )


def test_registry_dispatches_registered_tool():
    reg = ToolRegistry()
    reg.register(EchoTool())
    call = ToolCall(call_id="c1", action_type="echo", arguments={})
    result = reg.dispatch(call)
    assert result.ok is True
    assert result.action_type == "echo"
    assert result.observation.summary == "echo"


def test_registry_unknown_tool_returns_structured_error():
    reg = ToolRegistry()
    call = ToolCall(call_id="c2", action_type="nope", arguments={})
    result = reg.dispatch(call)
    assert result.ok is False
    assert "unknown tool" in result.error


def test_registry_names_sorted():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(SubmitAnswerTool())
    assert reg.names() == ["echo", "submit_answer"]


def test_registry_action_schema_text_stable_and_includes_names():
    reg = ToolRegistry.with_stable_defaults()
    text_a = reg.action_schema_text()
    text_b = reg.action_schema_text()
    assert text_a == text_b
    for name in [
        "explore_panorama",
        "navigate_to_object",
        "explore_seed",
        "explore_frontier",
        "submit_answer",
    ]:
        assert name in text_a


# ── SubmitAnswerTool ───────────────────────────────────────────────────────


def test_submit_answer_with_answer():
    tool = SubmitAnswerTool()
    call = ToolCall(
        call_id="c3",
        action_type="submit_answer",
        arguments={"answer": "the chair"},
    )
    result = tool.run(call)
    assert result.ok is True
    assert result.terminal is True
    assert "the chair" in result.observation.summary


def test_submit_answer_missing_answer():
    tool = SubmitAnswerTool()
    call = ToolCall(call_id="c4", action_type="submit_answer", arguments={})
    result = tool.run(call)
    assert result.ok is False
    assert result.terminal is True
    assert "requires an answer" in result.error


# ── NoopNavigationTool / defaults ─────────────────────────────────────────


@pytest.mark.parametrize(
    "action_type",
    [
        "explore_panorama",
        "navigate_to_object",
        "explore_seed",
        "explore_frontier",
    ],
)
def test_default_navigation_tools_dispatch_without_error(action_type):
    reg = ToolRegistry.with_stable_defaults()
    call = ToolCall(call_id=f"c_{action_type}", action_type=action_type, arguments={})
    result = reg.dispatch(call)
    assert result.ok is True
    assert result.terminal is False
    assert "unknown tool" not in result.error


def test_default_registry_has_no_fork_or_pixel():
    reg = ToolRegistry.with_stable_defaults()
    names = reg.names()
    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names
    assert "submit_answer" in names


def test_with_stable_defaults_names_exact():
    assert ToolRegistry.with_stable_defaults().names() == [
        "explore_frontier",
        "explore_panorama",
        "explore_seed",
        "navigate_to_object",
        "submit_answer",
    ]


def test_runtime_tool_is_abstract():
    with pytest.raises(TypeError):
        RuntimeTool()


def test_module_level_with_stable_defaults_alias():
    # Backward-compatible alias still works.
    reg = with_stable_defaults()
    assert isinstance(reg, ToolRegistry)
    assert "submit_answer" in reg.names()


def test_noop_navigation_reports_target_and_path_length():
    tool = NoopNavigationTool()
    call = ToolCall(
        call_id="c5",
        action_type="navigate_to_object",
        arguments={"object_name": "chair"},
    )
    result = tool.run(call)
    assert result.ok is True
    assert "path_length" in result.metrics
    assert "navigate_to_object" in result.observation.summary
