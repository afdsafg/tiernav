"""Default-path no-stub audit: dispatch every default tool with minimal calls."""

import pytest

from src.agent_evidence import TrajectoryEvidence
from src.tiernav_runtime.contracts import ToolCall, ToolResult
from src.tiernav_runtime.tools import ToolRegistry, build_real_tool_registry


def test_default_registry_excludes_stubs():
    registry = ToolRegistry.with_stable_defaults()
    names = set(registry.names())
    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names


@pytest.mark.parametrize("action_type", list(ToolRegistry.with_stable_defaults().names()))
def test_default_tools_dispatch_without_not_implemented(action_type):
    registry = ToolRegistry.with_stable_defaults()
    arguments = {"answer": "test"} if action_type == "submit_answer" else {}
    call = ToolCall(call_id="audit-1", action_type=action_type, arguments=arguments)
    result = registry.dispatch(call)
    assert isinstance(result, ToolResult)
    assert result.call_id == "audit-1"
    assert result.action_type == action_type
    # Name blacklist (test_default_registry_excludes_stubs) and behavior check
    # here are complementary: the name filter blocks known stubs by name, while
    # these assertions ensure dispatch actually succeeds rather than returning a
    # ToolResult that merely wraps a swallowed NotImplementedError.
    assert result.ok is True, f"{action_type} returned error: {result.error!r}"
    assert "NotImplementedError" not in result.error


class _FakeExecutor:
    def __init__(self) -> None:
        self._path_length = 0.5

    @property
    def path_length(self) -> float:
        return self._path_length

    def _ev(self) -> TrajectoryEvidence:
        return TrajectoryEvidence(
            subgoal="s", task_mode="m", progress="p", outcome="ok"
        )

    def explore_panorama(self, config=None):
        return self._ev()

    def navigate_to_object(self, object_name, view_idx=None):
        return self._ev()

    def explore_seed(self, seed_id):
        return self._ev()

    def explore_frontier(self, frontier_id):
        return self._ev()


def test_real_registry_excludes_stubs():
    registry = build_real_tool_registry(_FakeExecutor())
    names = set(registry.names())
    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names


@pytest.mark.parametrize("action_type", list(build_real_tool_registry(_FakeExecutor()).names()))
def test_real_registry_dispatches_without_not_implemented(action_type):
    registry = build_real_tool_registry(_FakeExecutor())
    arguments = {"answer": "test"} if action_type == "submit_answer" else {}
    if action_type == "navigate_to_object":
        arguments = {"object_name": "chair"}
    elif action_type == "explore_seed":
        arguments = {"seed_id": "s1"}
    elif action_type == "explore_frontier":
        arguments = {"frontier_id": "f1"}
    call = ToolCall(call_id="audit-real-1", action_type=action_type, arguments=arguments)
    result = registry.dispatch(call)
    assert isinstance(result, ToolResult)
    assert result.ok is True, f"{action_type} returned error: {result.error!r}"
    assert "NotImplementedError" not in result.error
