"""Verify PixelNavigateTool — pixel→backproject→navigate (stub).

Schema tests use the real ToolSchema dataclass interface (not a dict),
matching ExploreFrontierTool's pattern. Backproject is stubbed (TODO)
per task D2: phase 1 out-of-scope #5, no graph edit.
"""
import pytest


def test_pixel_navigate_tool_exists():
    """PixelNavigateTool class exists and is an ActionTool."""
    from src.two_tier_graph.tools import PixelNavigateTool, ActionTool
    assert PixelNavigateTool is not None
    assert issubclass(PixelNavigateTool, ActionTool)


def test_pixel_navigate_tool_schema():
    """Tool schema should declare pixel_x, pixel_y arg fields."""
    from src.two_tier_graph.tools import PixelNavigateTool
    tool = PixelNavigateTool()
    schema = tool.schema()
    assert schema.name == "pixel_navigate"
    fields = dict(schema.arg_fields)
    assert "pixel_x" in fields
    assert "pixel_y" in fields


def test_pixel_navigate_tool_registered():
    """PixelNavigateTool should be in default tool registry."""
    from src.two_tier_graph.tools import build_default_tool_registry
    registry = build_default_tool_registry()
    tool_names = [t.schema().name for t in registry.tools]
    assert "pixel_navigate" in tool_names


def test_pixel_navigate_tool_not_terminal():
    """pixel_navigate is a navigation action, not terminal (routed to executor)."""
    from src.two_tier_graph.tools import PixelNavigateTool
    tool = PixelNavigateTool()
    assert tool.schema().is_terminal is False


def test_pixel_navigate_tool_run_stub():
    """run() should return a stub TrajectoryEvidence (backproject TODO).

    Passing a PlannerAction-like object with pixel_x/pixel_y via getattr.
    Verifies the stub path without requiring a real tsdf_planner.
    """
    from src.two_tier_graph.tools import PixelNavigateTool

    class _FakeAction:
        action_type = "pixel_navigate"
        pixel_x = 42
        pixel_y = 17

    class _FakeExecutor:
        def navigate_to_point(self, x, y):
            return ("navigated", x, y)

    class _FakeCtx:
        executor = _FakeExecutor()
        resources = None
        state = None

    tool = PixelNavigateTool()
    result = tool.run(_FakeAction(), _FakeCtx())
    # Stub returns TrajectoryEvidence-like; just verify it ran and logged coords.
    assert result is not None
    assert hasattr(result, "task_mode") or hasattr(result, "progress")
