"""Tests for the LangGraph runtime graph skeleton.

Deterministic fakes only: no external services, no network. Exercises the
planned examples (tool-then-submit, round-budget fallback) plus extra
coverage for context compilation, memory query, tool-failure resilience,
and the default ContextCompiler on RuntimeServices.
"""
from __future__ import annotations

from typing import Optional

import pytest

from src.tiernav_runtime.context import ContextCompiler
from src.tiernav_runtime.contracts import (
    AblationConfig,
    EpisodeRequest,
    EpisodeState,
    Observation,
    PlannerDecision,
    RunSpec,
    ToolCall,
    ToolResult,
)
from src.tiernav_runtime.graph import (
    RuntimeGraphState,
    RuntimeServices,
    build_runtime_graph,
)
from src.tiernav_runtime.memory import MemoryService
from src.tiernav_runtime.policy import WorkflowPolicy
from src.tiernav_runtime.tools import RuntimeTool, ToolRegistry, with_stable_defaults


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakePlanner:
    """Deterministic planner that replays a scripted action sequence.

    Each call to decide returns the next PlannerDecision in ``script``. The
    final decision is repeated once the script is exhausted so the policy
    can route to a terminal node.
    """

    def __init__(self, script: list[PlannerDecision]) -> None:
        if not script:
            raise ValueError("script must be non-empty")
        self._script = list(script)
        self._i = 0

    def decide(self, prompt: str) -> PlannerDecision:
        decision = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return decision


def _request(prompt: str = "where is the mug?") -> EpisodeRequest:
    return EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt=prompt,
    )


def _spec(
    *,
    max_rounds: int = 10,
    max_steps: int = 50,
    ablation: Optional[AblationConfig] = None,
) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
        seed=0,
        max_rounds=max_rounds,
        max_steps=max_steps,
        ablation=ablation if ablation is not None else AblationConfig(),
    )


def _services(planner: FakePlanner) -> RuntimeServices:
    return RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
    )


def _run(services: RuntimeServices, spec: RunSpec, request: EpisodeRequest) -> EpisodeState:
    """Invoke the compiled graph and return the final EpisodeState."""
    app = build_runtime_graph(services)
    initial: RuntimeGraphState = {"spec": spec, "request": request, "state": None}
    result = app.invoke(initial)
    state = result["state"]
    assert isinstance(state, EpisodeState)
    return state


# ── Plan examples ─────────────────────────────────────────────────────────


def test_tool_then_submit_returns_answer():
    """FakePlanner: explore_frontier then submit_answer -> terminal, mug, step_index 1."""
    planner = FakePlanner(
        [
            PlannerDecision(
                action_type="explore_frontier",
                arguments={"frontier_id": "f1"},
            ),
            PlannerDecision(
                action_type="submit_answer",
                arguments={"answer": "mug"},
            ),
        ]
    )
    state = _run(_services(planner), _spec(), _request())

    assert state.terminal is True
    assert state.success is True
    assert state.answer == "mug"
    assert state.step_index == 1


def test_round_budget_fallback():
    """max_rounds=1 -> terminal, success False, failure_type round_budget."""
    planner = FakePlanner(
        [PlannerDecision(action_type="explore_frontier")]
    )
    state = _run(_services(planner), _spec(max_rounds=1), _request())

    assert state.terminal is True
    assert state.success is False
    assert state.answer == "unanswerable"
    assert state.failure_type == "round_budget"


# ── Extra coverage ────────────────────────────────────────────────────────


def test_context_sections_written_and_prompt_nonempty():
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "sofa"})]
    )
    state = _run(_services(planner), _spec(), _request())

    assert state.context_sections, "context_sections must be populated"
    names = [s.name for s in state.context_sections]
    assert "task_instruction" in names
    assert "action_schema" in names
    assert state.prompt, "prompt must be non-empty"


def test_memory_pack_present_when_active_memory_query_true():
    """active_memory_query=True -> memory_pack is populated before planning."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    # Seed memory so query() returns a non-empty pack.
    services = _services(planner)
    services.memory.update_from_observation(
        Observation(
            image_ids=["img-1"], summary="a mug on the counter", object_ids=["mug"]
        ),
        action_type="explore_panorama",
        round_index=0,
    )
    spec = _spec(ablation=AblationConfig(active_memory_query=True, spatial_memory=True))
    state = _run(services, spec, _request(prompt="mug"))

    assert state.memory_pack is not None
    assert state.memory_pack.query == "mug"
    assert state.memory_pack.summary, "memory_pack summary should be non-empty with a match"


def test_memory_pack_absent_when_active_memory_query_false():
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    spec = _spec(ablation=AblationConfig(active_memory_query=False, spatial_memory=False))
    state = _run(_services(planner), spec, _request())

    assert state.memory_pack is None


def test_tool_failure_from_registry_does_not_crash():
    """A tool returning a structured error must not raise; graph continues to fallback."""

    class AlwaysFailTool(RuntimeTool):
        name = "explore_frontier"

        def run(self, call: ToolCall) -> ToolResult:
            return ToolResult(
                call_id=call.call_id,
                action_type=call.action_type,
                ok=False,
                terminal=False,
                error="simulated tool failure",
            )

    registry = ToolRegistry()
    registry.register(AlwaysFailTool())
    registry.register(with_stable_defaults()._tools["submit_answer"])

    planner = FakePlanner(
        [
            PlannerDecision(action_type="explore_frontier"),
            PlannerDecision(action_type="submit_answer", arguments={"answer": "ok"}),
        ]
    )
    services = RuntimeServices(
        planner=planner,
        tools=registry,
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
    )
    # Should not raise; the failed tool yields a non-terminal error result and
    # the graph continues to the next round, which submits an answer.
    state = _run(services, _spec(), _request())

    assert state.terminal is True
    assert state.success is True
    assert state.answer == "ok"


def test_runtime_services_default_context_is_context_compiler():
    """RuntimeServices fills ContextCompiler by default when context=None."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        # context intentionally omitted
    )
    assert isinstance(services.context, ContextCompiler)


def test_runtime_services_explicit_context_preserved():
    """An explicitly supplied ContextCompiler is not replaced."""
    planner = FakePlanner(
        [PlannerDecision(action_type="submit_answer", arguments={"answer": "x"})]
    )
    custom = ContextCompiler()
    services = RuntimeServices(
        planner=planner,
        tools=with_stable_defaults(),
        memory=MemoryService(enabled=True),
        policy=WorkflowPolicy(),
        context=custom,
    )
    assert services.context is custom


def test_graph_state_typeddict_has_required_keys():
    """RuntimeGraphState declares spec/request/state/policy/prompt at least."""
    # TypedDict __annotations__ reveal the declared keys.
    ann = RuntimeGraphState.__annotations__
    for key in ("spec", "request", "state", "policy", "prompt"):
        assert key in ann, f"RuntimeGraphState missing key: {key}"
