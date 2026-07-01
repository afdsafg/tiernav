"""Tests for planner adapter and stable tool registry."""
from __future__ import annotations

import pytest

from src.agent_evidence import TrajectoryEvidence
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
    build_real_tool_registry,
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


class BoomTool(RuntimeTool):
    name = "boom"

    def run(self, call: ToolCall) -> ToolResult:  # noqa: D401
        raise RuntimeError("boom")


def test_registry_dispatch_catches_tool_exception():
    reg = ToolRegistry()
    reg.register(BoomTool())
    call = ToolCall(call_id="c_boom", action_type="boom", arguments={})
    result = reg.dispatch(call)
    assert result.ok is False
    assert result.terminal is False
    assert "RuntimeError" in result.error
    assert "boom" in result.error


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


# ── Real tool registry wrapping Executor ──────────────────────────────────


class FakeExecutor:
    """Quacks like Executor: 4 navigation methods + path_length property.

    Records calls for assertion and returns real TrajectoryEvidence so the
    evidence->ToolResult conversion is exercised against the real dataclass.
    """

    def __init__(self, path_length: float = 1.25) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._path_length = path_length

    @property
    def path_length(self) -> float:
        return self._path_length

    def _record(self, name: str, args: tuple, kwargs: dict) -> TrajectoryEvidence:
        self.calls.append((name, args, kwargs))
        return TrajectoryEvidence(
            subgoal="fake-subgoal",
            task_mode=name,
            progress="reached",
            salient=["red chair"],
            outcome="object_found",
            gd_quality="ok",
            key_frames=["frame_000", "frame_007"],
            current_image_b64=None,
            room_id=3,
            objects_nearby=["chair", "table"],
        )

    def explore_panorama(self, config=None) -> TrajectoryEvidence:
        return self._record("explore_panorama", (), {"config": config})

    def navigate_to_object(
        self, object_name: str, view_idx=None
    ) -> TrajectoryEvidence:
        return self._record(
            "navigate_to_object", (object_name,), {"view_idx": view_idx}
        )

    def explore_seed(self, seed_id: str) -> TrajectoryEvidence:
        return self._record("explore_seed", (seed_id,), {})

    def explore_frontier(self, frontier_id: str) -> TrajectoryEvidence:
        return self._record("explore_frontier", (frontier_id,), {})


def _evidence() -> TrajectoryEvidence:
    return TrajectoryEvidence(
        subgoal="s",
        task_mode="navigate_to_object",
        progress="p",
        outcome="object_found",
        gd_quality="ok",
        key_frames=["k1"],
        room_id=2,
        objects_nearby=["o1"],
    )


def test_runtime_tools_wrap_executor_methods():
    registry = build_real_tool_registry(FakeExecutor())
    names = registry.names()
    for required in (
        "explore_panorama",
        "navigate_to_object",
        "explore_seed",
        "explore_frontier",
        "submit_answer",
    ):
        assert required in names
    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names


def test_real_registry_explore_panorama_dispatches_and_builds_result():
    fake = FakeExecutor(path_length=2.5)
    reg = build_real_tool_registry(fake)
    call = ToolCall(call_id="p1", action_type="explore_panorama", arguments={})
    result = reg.dispatch(call)
    assert fake.calls == [("explore_panorama", (), {"config": None})]
    assert result.ok is True
    assert result.terminal is False
    assert result.metrics["path_length"] == pytest.approx(2.5)
    obs = result.observation
    assert obs.summary  # non-empty
    # Legacy priority: progress or outcome.
    assert obs.summary == "reached"
    assert obs.image_ids == ["frame_000", "frame_007"]
    assert obs.object_ids == ["chair", "table"]
    assert obs.room_id == "3"
    assert obs.pose == {}
    assert obs.raw["outcome"] == "object_found"
    assert obs.raw["gd_quality"] == "ok"
    assert obs.raw["subgoal"] == "fake-subgoal"
    assert obs.raw["progress"] == "reached"
    assert obs.raw["salient"] == ["red chair"]
    assert obs.raw["path_length"] == pytest.approx(2.5)
    assert obs.raw["path_delta"] == pytest.approx(0.0)


def test_real_registry_navigate_to_object_passes_args():
    fake = FakeExecutor()
    reg = build_real_tool_registry(fake)
    call = ToolCall(
        call_id="n1",
        action_type="navigate_to_object",
        arguments={"object_name": "chair", "view_idx": 4},
    )
    result = reg.dispatch(call)
    assert result.ok is True
    assert fake.calls == [("navigate_to_object", ("chair",), {"view_idx": 4})]
    assert result.metrics["path_length"] == pytest.approx(1.25)


def test_real_registry_explore_seed_passes_args():
    fake = FakeExecutor()
    reg = build_real_tool_registry(fake)
    call = ToolCall(
        call_id="s1",
        action_type="explore_seed",
        arguments={"seed_id": "seed_12"},
    )
    result = reg.dispatch(call)
    assert result.ok is True
    assert fake.calls == [("explore_seed", ("seed_12",), {})]


def test_real_registry_explore_frontier_passes_args():
    fake = FakeExecutor()
    reg = build_real_tool_registry(fake)
    call = ToolCall(
        call_id="f1",
        action_type="explore_frontier",
        arguments={"frontier_id": "fr_9"},
    )
    result = reg.dispatch(call)
    assert result.ok is True
    assert fake.calls == [("explore_frontier", ("fr_9",), {})]


def test_real_registry_marks_failed_executor_outcome_as_not_ok():
    class FailedFrontierExecutor(FakeExecutor):
        def explore_frontier(self, frontier_id: str) -> TrajectoryEvidence:
            self.calls.append(("explore_frontier", (frontier_id,), {}))
            return TrajectoryEvidence(
                subgoal=f"Navigate to frontier {frontier_id}",
                task_mode="explore_frontier",
                progress=f"Frontier {frontier_id} not found",
                outcome="target_not_reached",
                gd_quality="no_detection",
            )

    fake = FailedFrontierExecutor()
    reg = build_real_tool_registry(fake)
    call = ToolCall(
        call_id="f_missing",
        action_type="explore_frontier",
        arguments={"frontier_id": "0"},
    )
    result = reg.dispatch(call)

    assert fake.calls == [("explore_frontier", ("0",), {})]
    assert result.ok is False
    assert result.terminal is False
    assert "target_not_reached" in result.error
    assert "Frontier 0 not found" in result.error
    assert result.observation.raw["outcome"] == "target_not_reached"


def test_real_registry_executor_error_returns_structured_failure():
    class BoomExecutor(FakeExecutor):
        def navigate_to_object(self, object_name, view_idx=None):
            raise RuntimeError("no path")

    reg = build_real_tool_registry(BoomExecutor())
    call = ToolCall(
        call_id="e1",
        action_type="navigate_to_object",
        arguments={"object_name": "chair"},
    )
    result = reg.dispatch(call)
    assert result.ok is False
    assert result.terminal is False
    assert "RuntimeError" in result.error
    assert "no path" in result.error


def test_real_registry_submit_answer_is_terminal_and_records():
    reg = build_real_tool_registry(FakeExecutor())
    call = ToolCall(
        call_id="a1",
        action_type="submit_answer",
        arguments={"answer": "the red chair"},
    )
    result = reg.dispatch(call)
    assert result.ok is True
    assert result.terminal is True
    assert "the red chair" in result.observation.summary


def test_real_registry_room_id_none_when_unset():
    class NoRoomExecutor(FakeExecutor):
        def _record(self, name, args, kwargs):
            ev = super()._record(name, args, kwargs)
            ev.room_id = -1
            return ev

    reg = build_real_tool_registry(NoRoomExecutor())
    call = ToolCall(call_id="p2", action_type="explore_panorama", arguments={})
    result = reg.dispatch(call)
    assert result.ok is True
    assert result.observation.room_id is None


def test_evidence_to_observation_preserves_progress_and_salient_in_raw():
    """raw must carry progress/salient so legacy consumers stay aligned."""
    from src.tiernav_runtime.tools import _evidence_to_observation

    ev = TrajectoryEvidence(
        subgoal="sg",
        task_mode="navigate_to_object",
        progress="moved to chair",
        salient=["red chair", "window"],
        outcome="object_found",
        gd_quality="ok",
        key_frames=["k1"],
        room_id=2,
        objects_nearby=["o1"],
    )
    obs = _evidence_to_observation(ev)
    assert obs.raw["progress"] == "moved to chair"
    assert obs.raw["salient"] == ["red chair", "window"]
    # summary prefers progress over outcome (legacy priority).
    assert obs.summary == "moved to chair"


def test_evidence_to_observation_summary_falls_back_to_outcome_when_progress_empty():
    """When progress is empty, summary falls back to outcome."""
    from src.tiernav_runtime.tools import _evidence_to_observation

    ev = TrajectoryEvidence(
        subgoal="sg",
        task_mode="navigate_to_object",
        progress="",
        salient=[],
        outcome="object_found",
        gd_quality="ok",
        key_frames=["k1"],
        room_id=2,
        objects_nearby=["o1"],
    )
    obs = _evidence_to_observation(ev)
    assert obs.summary == "object_found"
    assert obs.raw["progress"] == ""
    assert obs.raw["salient"] == []
