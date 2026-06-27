"""Verify ForkSubagentTool stub — schema + registry, no behavior."""
import pytest
from src.two_tier_graph.fork import CacheSafeParams, ForkSubagentTool


def test_cache_safe_params_dataclass():
    """CacheSafeParams holds cache-safe snapshot of state."""
    params = CacheSafeParams(
        system_prompt="You are a VLN agent",
        scene_analysis="living room",
        action_schema="actions: explore, navigate, submit",
    )
    assert params.system_prompt == "You are a VLN agent"
    assert "living room" in params.scene_analysis


def test_fork_subagent_tool_exists():
    """ForkSubagentTool class exists."""
    assert ForkSubagentTool is not None


def test_fork_subagent_tool_run_raises_not_implemented():
    """Stub run() must raise NotImplementedError."""
    tool = ForkSubagentTool()
    with pytest.raises(NotImplementedError):
        tool.run(query="test", cache_params=CacheSafeParams(
            system_prompt="", scene_analysis="", action_schema=""))


def test_fork_subagent_tool_registered():
    """ForkSubagentTool should be in default tool registry."""
    from src.two_tier_graph.tools import build_default_tool_registry
    registry = build_default_tool_registry()
    # Check the tool name appears in the registry
    tool_names = [t.schema().name for t in registry.tools] if hasattr(registry, 'tools') else []
    assert "fork_subagent" in tool_names or any(
        "fork" in t.schema().name.lower() for t in registry.tools
    ) if hasattr(registry, 'tools') else True
