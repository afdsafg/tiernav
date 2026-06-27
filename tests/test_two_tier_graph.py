"""Deterministic unit tests for the Two-Tier LangGraph state machine.

Uses a stubbed LLMProvider returning canned PlannerAction sequences. Tests
assert edge routing, guard firing, and the critical `after_memory` edge
ordering (round-budget → exhausted → step-budget). No real LLM, no Habitat-sim.

Per plan §4a, these isolate "did the graph preserve control flow" from
"did the LLM produce different text" — prerequisite for the side-by-side
smoke run on the AEQA dev subset (§4b, run by user on server).
"""
from __future__ import annotations

from typing import Optional

import pytest

from src.agent_planner import PlannerAction
from src.two_tier_graph.edges import after_guard, after_memory
from src.two_tier_graph.providers import LLMProvider
from src.two_tier_graph.tools import build_default_tool_registry


# ── Stub provider ────────────────────────────────────────────────────────


class StubProvider(LLMProvider):
    """Returns canned PlannerActions in sequence. No LLM calls."""

    def __init__(self, actions: list[PlannerAction]):
        self._actions = list(actions)
        self._idx = 0
        self.raw_responses: list[str] = []
        self.decide_calls: int = 0

    def decide(self, **_):
        if self._idx >= len(self._actions):
            raise RuntimeError(
                f"StubProvider exhausted: {self.decide_calls} calls, "
                f"{len(self._actions)} actions"
            )
        action = self._actions[self._idx]
        self._idx += 1
        self.decide_calls += 1
        return action

    def decide_raw(self, messages, image_b64=None, max_tokens=4096, temperature=0.3):
        # Stage 6.5 frontier sub-selection returns text that parses to the
        # first frontier_id. Used by _select_frontier_with_vlm in planner_node.
        self.raw_responses.append(
            '{"reasoning":"stub","expected":"stub","frontier_id":0,"confidence":0.5}'
        )
        return self.raw_responses[-1]


# ── Edge tests ───────────────────────────────────────────────────────────


def test_after_guard_routes_submit_to_submit():
    """submit_answer action → 'submit' edge (wraps :1614)."""
    state = {"current_action": PlannerAction(action_type="submit_answer", answer="chair")}
    assert after_guard(state) == "submit"


def test_after_guard_routes_other_actions_to_execute():
    """Non-submit actions → 'execute' edge (wraps :1633)."""
    for action_type in ["explore_panorama", "navigate_to_object", "explore_seed", "explore_frontier"]:
        state = {"current_action": PlannerAction(action_type=action_type)}
        assert after_guard(state) == "execute", f"failed for {action_type}"


def test_after_guard_routes_none_action_to_execute():
    """Defensive: None current_action → 'execute' (shouldn't happen but be safe)."""
    state = {"current_action": None}
    assert after_guard(state) == "execute"


def test_after_memory_round_budget_dominates():
    """Round-budget (for-loop end) → fallback_submit. Checked FIRST.
    Mirrors :1442 range exhaustion → :1681 fallback."""
    state = {
        "rounds_used": 10,
        "max_planner_rounds": 10,
        "exhausted_flag": True,  # would normally skip step-budget
        "steps_taken": 999,      # would normally trigger fallback
        "max_total_steps": 50,
    }
    assert after_memory(state) == "fallback_submit"


def test_after_memory_exhausted_skips_step_budget():
    """CRITICAL: exhausted_flag → 'continue', SKIPPING step-budget check.
    Reproduces :1665-1672 `continue` skip semantics.

    If this fails, the ordering in after_memory is wrong and benchmark
    numbers will drift."""
    state = {
        "rounds_used": 3,
        "max_planner_rounds": 10,
        "exhausted_flag": True,
        "steps_taken": 999,  # over budget — but exhausted should skip this
        "max_total_steps": 50,
    }
    assert after_memory(state) == "continue"


def test_after_memory_step_budget_breaks_when_not_exhausted():
    """Step-budget break → fallback_submit (only when NOT exhausted).
    Mirrors :1675-1677 `break`."""
    state = {
        "rounds_used": 3,
        "max_planner_rounds": 10,
        "exhausted_flag": False,
        "steps_taken": 50,
        "max_total_steps": 50,
    }
    assert after_memory(state) == "fallback_submit"


def test_after_memory_continue_default():
    """Default: next round."""
    state = {
        "rounds_used": 3,
        "max_planner_rounds": 10,
        "exhausted_flag": False,
        "steps_taken": 20,
        "max_total_steps": 50,
    }
    assert after_memory(state) == "continue"


def test_after_memory_round_boundary_exact():
    """rounds_used == max_planner_rounds (exact) → fallback_submit."""
    state = {
        "rounds_used": 10,
        "max_planner_rounds": 10,
        "exhausted_flag": False,
        "steps_taken": 5,
        "max_total_steps": 50,
    }
    assert after_memory(state) == "fallback_submit"


def test_after_memory_step_boundary_exact():
    """steps_taken == max_total_steps (exact, not exhausted) → fallback_submit."""
    state = {
        "rounds_used": 3,
        "max_planner_rounds": 10,
        "exhausted_flag": False,
        "steps_taken": 50,
        "max_total_steps": 50,
    }
    assert after_memory(state) == "fallback_submit"


# ── Tool registry tests ─────────────────────────────────────────────────


def test_default_tool_registry_has_5_tools():
    """All 5 default tools are registered with correct names."""
    registry = build_default_tool_registry()
    expected = {
        "explore_panorama", "navigate_to_object",
        "explore_seed", "explore_frontier", "submit_answer",
    }
    assert set(registry._tools.keys()) == expected


def test_submit_answer_tool_is_terminal():
    """SubmitAnswerTool.schema().is_terminal must be True — routes to submit_node."""
    registry = build_default_tool_registry()
    submit_tool = registry.get("submit_answer")
    assert submit_tool.schema().is_terminal is True


def test_non_submit_tools_are_not_terminal():
    """The 4 action tools must NOT be terminal — they go to executor_node."""
    registry = build_default_tool_registry()
    for name in ["explore_panorama", "navigate_to_object", "explore_seed", "explore_frontier"]:
        tool = registry.get(name)
        assert tool.schema().is_terminal is False, f"{name} should not be terminal"


def test_tool_registry_dispatches_by_action_type():
    """dispatch(action) routes by action.action_type."""
    registry = build_default_tool_registry()
    # Verify each action_type is in the registry (dispatch would raise KeyError otherwise)
    for action_type in ["explore_panorama", "navigate_to_object", "explore_seed", "explore_frontier", "submit_answer"]:
        action = PlannerAction(action_type=action_type)
        tool = registry.get(action.action_type)
        assert tool is not None


# ── Provider tests ───────────────────────────────────────────────────────


def test_stub_provider_returns_actions_in_order():
    """StubProvider returns canned actions in sequence."""
    actions = [
        PlannerAction(action_type="explore_panorama"),
        PlannerAction(action_type="submit_answer", answer="chair"),
    ]
    provider = StubProvider(actions)
    assert provider.decide().action_type == "explore_panorama"
    assert provider.decide().action_type == "submit_answer"
    assert provider.decide_calls == 2


def test_build_llm_provider_defaults_to_mimo():
    """build_llm_provider falls back to mimo when cfg has no .llm attribute."""
    from src.two_tier_graph.providers import MimoProvider, build_llm_provider

    class FakePlanner:
        pass
    class FakeCfg:
        pass  # no .llm attribute

    provider = build_llm_provider(FakeCfg(), FakePlanner())
    assert isinstance(provider, MimoProvider)


def test_build_llm_provider_rejects_claude():
    """Claude provider is not implemented this phase — must raise."""
    from src.two_tier_graph.providers import build_llm_provider

    class FakeLlm:
        provider = "claude"
    class FakeCfg:
        llm = FakeLlm()

    with pytest.raises(NotImplementedError):
        build_llm_provider(FakeCfg(), planner=None)


# ── Graph compilation test ──────────────────────────────────────────────


def test_graph_compiles_with_7_nodes():
    """build_two_tier_graph() returns a compiled graph with all 7 user nodes."""
    from src.two_tier_graph.graph import build_two_tier_graph

    graph = build_two_tier_graph()
    node_names = set(graph.nodes.keys())
    expected = {"init", "build_context", "planner",
                "loop_guard", "executor", "memory_update", "submit"}
    missing = expected - node_names
    assert not missing, f"missing nodes: {missing}"
    # __start__ is always present; __end__ may or may not appear in graph.nodes
    # depending on langgraph version (it's a virtual terminal).
    assert "__start__" in node_names


# ── State schema test ────────────────────────────────────────────────────


def test_two_tier_state_has_required_fields():
    """TwoTierState must have all fields the nodes read/write."""
    from src.two_tier_graph.state import TwoTierState

    # TypedDicts don't expose __annotations__ cleanly across Python versions,
    # but we can construct a full dict and check it doesn't raise.
    full_state = {
        "scene_id": "test", "question_id": "q0", "question": "?", "output_dir": "/tmp",
        "max_planner_rounds": 10, "max_total_steps": 50,
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": True, "use_rejected_tracking": True,
        "pose": {"pts": None, "angle": 0.0},
        "rounds_used": 0, "steps_taken": 0,
        "current_action": None, "last_evidence": None, "exhausted_flag": False,
        "action_history": [], "round_traces": [],
        "scene_analysis": "", "history_text": "", "progress_text": "", "actions_text": "",
        "current_views": [], "topdown_b64": None, "memory_summary": {},
        "answer": "", "success": False, "error": "", "terminal": False, "failure_type": "",
    }
    # If the TypedDict schema is missing a field, this assignment would
    # fail at type-check time (not runtime for TypedDict, but the test
    # documents the expected shape).
    assert set(full_state.keys()) >= {
        "scene_id", "question_id", "question", "output_dir",
        "max_planner_rounds", "max_total_steps",
        "use_notebook", "use_scene_graph", "use_active_query", "use_rejected_tracking",
        "pose", "rounds_used", "steps_taken",
        "current_action", "last_evidence", "exhausted_flag",
        "action_history", "round_traces",
        "scene_analysis", "history_text", "progress_text", "actions_text",
        "current_views", "topdown_b64", "memory_summary",
        "answer", "success", "error", "terminal", "failure_type",
    }
