"""Stable tool registry and runtime tool contracts.

No external services, no LangGraph. Tools are deterministic and safe to
invoke in tests and replay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .contracts import Observation, ToolCall, ToolResult


class RuntimeTool(ABC):
    """Abstract base for runtime tools.

    Subclasses set ``name`` and optionally ``terminal``, and implement
    :meth:`run`. Tools must not raise on well-formed calls; they return a
    structured :class:`ToolResult` instead.
    """

    name: ClassVar[str] = ""
    terminal: ClassVar[bool] = False

    @abstractmethod
    def run(self, call: ToolCall) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


class SubmitAnswerTool(RuntimeTool):
    """Terminal tool that records the planner's final answer."""

    name: ClassVar[str] = "submit_answer"
    terminal: ClassVar[bool] = True

    def run(self, call: ToolCall) -> ToolResult:
        answer = str(call.arguments.get("answer", "") or "")
        if not answer:
            return ToolResult(
                call_id=call.call_id,
                action_type=call.action_type,
                ok=False,
                terminal=True,
                error="submit_answer requires an answer",
            )
        return ToolResult(
            call_id=call.call_id,
            action_type=call.action_type,
            ok=True,
            terminal=True,
            observation=Observation(summary=f"submitted answer: {answer}"),
        )


class NoopNavigationTool(RuntimeTool):
    """Deterministic no-op tool for navigation/exploration actions.

    Used for the default navigation action types. Does not raise; reports
    the dispatched target and a zero path length so downstream metrics
    remain well-formed.
    """

    name: ClassVar[str] = "noop_navigation"

    def run(self, call: ToolCall) -> ToolResult:
        target = call.arguments.get("object_name") or call.arguments.get(
            "seed_id"
        ) or call.arguments.get("frontier_id") or ""
        summary = f"executed {call.action_type}"
        if target:
            summary += f" toward {target}"
        return ToolResult(
            call_id=call.call_id,
            action_type=call.action_type,
            ok=True,
            terminal=False,
            observation=Observation(summary=summary),
            metrics={"path_length": 0.0},
        )


class ToolRegistry:
    """Registry mapping action_type to a RuntimeTool."""

    def __init__(self) -> None:
        self._tools: dict[str, RuntimeTool] = {}

    def register(self, tool: RuntimeTool) -> None:
        if not tool.name:
            raise ValueError("tool must define a non-empty name")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.action_type)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                action_type=call.action_type,
                ok=False,
                terminal=False,
                error=f"unknown tool: {call.action_type}",
            )
        return tool.run(call)

    def action_schema_text(self) -> str:
        lines = ["Available tools:"]
        for name in self.names():
            tool = self._tools[name]
            lines.append(f"- {name}: terminal={tool.terminal}")
        return "\n".join(lines)

    @classmethod
    def with_stable_defaults(cls) -> "ToolRegistry":
        """Return a registry with the stable default tool set.

        Registers the four navigation actions (backed by NoopNavigationTool) and
        submit_answer. Does not register fork_subagent or pixel_navigate, and
        contains no stubs that raise NotImplementedError.
        """
        registry = cls()
        for action_type in _DEFAULT_NAVIGATION_ACTIONS:
            # Register a distinct tool instance per action_type so name lookups
            # resolve to the dispatched action. We override name per instance.
            tool = NoopNavigationTool()
            tool.name = action_type  # type: ignore[misc]
            registry.register(tool)
        registry.register(SubmitAnswerTool())
        return registry


_DEFAULT_NAVIGATION_ACTIONS = (
    "explore_panorama",
    "navigate_to_object",
    "explore_seed",
    "explore_frontier",
)


def with_stable_defaults() -> ToolRegistry:
    """Backward-compatible alias for :meth:`ToolRegistry.with_stable_defaults`."""
    return ToolRegistry.with_stable_defaults()
