"""ToolRegistry + ActionTool — extensible Executor dispatch.

Replaces the hard-coded if-chain at agent_executor.py:291. Each tool wraps an
existing `Executor` method 1:1 so behavior is identical. Future tools (e.g.
PixelNavigateTool for pixel→backproject→navigate) register via
`registry.register(...)` with NO graph edit — their `schema().prompt_description`
auto-appends to the action-space prompt via `actions_prompt_text`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.agent_evidence import TrajectoryEvidence


@dataclass
class ToolSchema:
    """Declarative schema for an action tool."""

    name: str                                       # e.g. "navigate_to_object"
    arg_fields: list[tuple[str, str]] = field(default_factory=list)
                                                    # [("snapshot_id","str"), ("object_name","str")]
    prompt_description: str = ""                    # injected into the actions prompt
    is_terminal: bool = False                       # submit_answer → True (routed to submit_node)


class ActionTool(ABC):
    """Base class for executor tools. `run` returns a TrajectoryEvidence."""

    @abstractmethod
    def schema(self) -> ToolSchema:
        ...

    @abstractmethod
    def run(self, action, ctx: "ToolContext") -> TrajectoryEvidence:
        ...


@dataclass
class ToolContext:
    """Context passed to each tool at dispatch time."""

    executor: object            # src.agent_executor.Executor
    resources: object           # Resources
    state: object               # TwoTierState (dict-like)


class ToolRegistry:
    """Registry of ActionTools, dispatchable by action_type.

    The 5 default tools wrap existing Executor methods (agent_executor.py:89-287)
    so behavior is byte-identical to the legacy `Executor.execute_action` if-chain
    at :291-319.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ActionTool] = {}

    def register(self, tool: ActionTool) -> None:
        name = tool.schema().name
        self._tools[name] = tool

    def get(self, name: str) -> ActionTool:
        return self._tools[name]  # KeyError → executor_node logs "unknown action"

    def dispatch(self, action, ctx: ToolContext) -> TrajectoryEvidence:
        tool = self.get(action.action_type)
        return tool.run(action, ctx)

    @property
    def tools(self) -> list[ActionTool]:
        return list(self._tools.values())

    def actions_prompt_text(self) -> str:
        """Concatenate each tool's prompt_description.

        NOTE: this phase keeps the legacy `_build_actions` body verbatim
        (agent_workflow.py:1360-1404) for byte-identical prompts. This method
        is provided for future use when the action-space prompt is migrated
        to be registry-driven (out of scope per plan §8 #6).
        """
        return "\n".join(t.schema().prompt_description for t in self._tools.values())


# ── 5 default tools (this phase) ─────────────────────────────────────────


class ExplorePanoramaTool(ActionTool):
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="explore_panorama",
            arg_fields=[],
            prompt_description="1. explore_panorama: re-orient with full panorama",
            is_terminal=False,
        )

    def run(self, action, ctx: ToolContext) -> TrajectoryEvidence:
        return ctx.executor.explore_panorama()


class NavigateToObjectTool(ActionTool):
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="navigate_to_object",
            arg_fields=[("snapshot_id", "str"), ("object_name", "str")],
            prompt_description=(
                '2. navigate_to_object: {"reasoning":"why","expected":"what this should verify",'
                '"action":"navigate_to_object","arguments":{"snapshot_id":"stepN_viewM",'
                '"object_name":"visible object"}}'
            ),
            is_terminal=False,
        )

    def run(self, action, ctx: ToolContext) -> TrajectoryEvidence:
        return ctx.executor.navigate_to_object(action.object_name, action.view_idx)


class ExploreSeedTool(ActionTool):
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="explore_seed",
            arg_fields=[("seed_id", "str")],
            prompt_description="3. explore_seed: navigate to a seed viewpoint",
            is_terminal=False,
        )

    def run(self, action, ctx: ToolContext) -> TrajectoryEvidence:
        return ctx.executor.explore_seed(action.seed_id)


class ExploreFrontierTool(ActionTool):
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="explore_frontier",
            arg_fields=[("frontier_id", "str")],
            prompt_description="3. explore_frontier <id>: navigate to one of the current frontiers",
            is_terminal=False,
        )

    def run(self, action, ctx: ToolContext) -> TrajectoryEvidence:
        return ctx.executor.explore_frontier(action.frontier_id)


class SubmitAnswerTool(ActionTool):
    """Registered so the action-space prompt lists submit_answer, but
    `is_terminal=True` routes it to `submit_node` via the `after_guard` edge —
    never to `executor_node`. `run()` is unreachable."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="submit_answer",
            arg_fields=[("snapshot_id", "str"), ("answer", "str")],
            prompt_description=(
                '4. submit_answer: {"reasoning":"why","action":"submit_answer",'
                '"arguments":{"snapshot_id":"stepN_viewM","answer":"final answer"}}'
            ),
            is_terminal=True,
        )

    def run(self, action, ctx: ToolContext) -> TrajectoryEvidence:
        # Unreachable: after_guard edge routes submit_answer to submit_node.
        # Kept for completeness; matches Executor.execute_action's submit_answer
        # branch (agent_executor.py:300-309) in case it's ever called directly.
        return TrajectoryEvidence(
            subgoal="Submit answer",
            task_mode="submit_answer",
            progress=f"Answer: {action.answer}",
            salient=[action.answer or ""],
            outcome="answer_submitted",
            room_id=-1,
            objects_nearby=[],
        )


def build_default_tool_registry() -> ToolRegistry:
    """Build a ToolRegistry pre-populated with the 5 default tools.

    Each tool wraps an existing `Executor` method 1:1, so dispatch through the
    registry is byte-identical to the legacy `Executor.execute_action` if-chain.
    """
    registry = ToolRegistry()
    registry.register(ExplorePanoramaTool())
    registry.register(NavigateToObjectTool())
    registry.register(ExploreSeedTool())
    registry.register(ExploreFrontierTool())
    registry.register(SubmitAnswerTool())
    from src.two_tier_graph.fork import ForkSubagentTool  # lazy: avoids circular import
    registry.register(ForkSubagentTool())
    return registry
