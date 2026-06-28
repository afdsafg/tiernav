"""Default-path no-stub audit: dispatch every default tool with minimal calls."""

import pytest

from src.tiernav_runtime.contracts import ToolCall, ToolResult
from src.tiernav_runtime.tools import ToolRegistry


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
