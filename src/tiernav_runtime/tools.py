"""Stable tool registry and runtime tool contracts.

No external services, no LangGraph. Tools are deterministic and safe to
invoke in tests and replay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

from .contracts import Observation, ToolCall, ToolResult


class RuntimeTool(ABC):
    """Abstract base for runtime tools.

    Subclasses set ``name`` and optionally ``terminal``, and implement
    :meth:`run`. Tools must not raise on well-formed calls; they return a
    structured :class:`ToolResult` instead.
    """

    name: ClassVar[str] = ""
    terminal: ClassVar[bool] = False
    # Human-readable description of required/optional arguments, used by
    # action_schema_text so the planner VLM knows what fields to emit.
    arg_schema: ClassVar[str] = ""

    @abstractmethod
    def run(self, call: ToolCall) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


class SubmitAnswerTool(RuntimeTool):
    """Terminal tool that records the planner's final answer."""

    name: ClassVar[str] = "submit_answer"
    terminal: ClassVar[bool] = True
    arg_schema: ClassVar[str] = "required: answer (str)"

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
        try:
            return tool.run(call)
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                action_type=call.action_type,
                ok=False,
                terminal=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def action_schema_text(self) -> str:
        lines = ["Available tools (arguments go in the JSON 'arguments' object):"]
        for name in self.names():
            tool = self._tools[name]
            args = tool.arg_schema or "none"
            lines.append(f"- {name}: {args} [terminal={tool.terminal}]")
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


# ── Executor-backed production tools ───────────────────────────────────────


@runtime_checkable
class _ExecutorLike(Protocol):
    """Structural type for objects that quack like ``Executor``.

    Only the surface used by the production tool registry is required. The
    real ``Executor`` (src/agent_executor.py) satisfies this; tests pass a
    fake with the same methods. Intentionally not imported here to avoid a
    habitat/TSDF dependency at module import time.
    """

    @property
    def path_length(self) -> float: ...

    def explore_panorama(self, config: Any = None) -> Any: ...

    def navigate_to_object(
        self, object_name: str, view_idx: Any = None
    ) -> Any: ...

    def explore_seed(self, seed_id: str) -> Any: ...

    def explore_frontier(self, frontier_id: str) -> Any: ...


def _evidence_to_observation(evidence: Any) -> Observation:
    """Build a JSON-safe :class:`Observation` from a ``TrajectoryEvidence``.

    ``pose`` is left empty: the runtime environment service (Task 3) tracks
    pose out-of-band and threads it through ``EpisodeState``. Task 7 will
    populate it here once the pose channel is wired through dispatch.
    """
    summary = getattr(evidence, "progress", "") or getattr(evidence, "outcome", "")
    room_id = getattr(evidence, "room_id", -1)
    return Observation(
        summary=str(summary),
        image_ids=list(getattr(evidence, "key_frames", []) or []),
        object_ids=list(getattr(evidence, "objects_nearby", []) or []),
        room_id=str(room_id) if room_id is not None and room_id >= 0 else None,
        pose={},
        raw={
            "outcome": str(getattr(evidence, "outcome", "") or ""),
            "gd_quality": str(getattr(evidence, "gd_quality", "") or ""),
            "subgoal": str(getattr(evidence, "subgoal", "") or ""),
            "progress": str(getattr(evidence, "progress", "") or ""),
            "salient": list(getattr(evidence, "salient", []) or []),
        },
    )


def _evidence_to_result(
    call: ToolCall,
    evidence: Any,
    path_length: float,
    terminal: bool = False,
    path_delta: float = 0.0,
) -> ToolResult:
    observation = _evidence_to_observation(evidence)
    observation.raw["action_type"] = call.action_type
    observation.raw["path_length"] = float(path_length)
    observation.raw["path_delta"] = float(path_delta)
    outcome = observation.raw.get("outcome", "")
    failed_outcomes = {"target_not_reached", "detection_failed", "error"}
    ok = outcome not in failed_outcomes
    error = ""
    if not ok:
        progress = observation.raw.get("progress", "")
        error = outcome if not progress else f"{outcome}: {progress}"
    return ToolResult(
        call_id=call.call_id,
        action_type=call.action_type,
        ok=ok,
        terminal=terminal,
        observation=observation,
        error=error,
        metrics={"path_length": float(path_length)},
    )


def _error_result(call: ToolCall, exc: BaseException) -> ToolResult:
    import traceback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return ToolResult(
        call_id=call.call_id,
        action_type=call.action_type,
        ok=False,
        terminal=False,
        error=f"{type(exc).__name__}: {exc}\n{tb}",
    )


class _ExecutorNavigationTool(RuntimeTool):
    """Base for executor-backed navigation tools.

    Subclasses set ``name`` and implement :meth:`invoke` to call the
    appropriate executor method and return its ``TrajectoryEvidence``.
    """

    name: ClassVar[str] = ""
    terminal: ClassVar[bool] = False

    def __init__(self, executor: _ExecutorLike) -> None:
        self._executor = executor

    @abstractmethod
    def invoke(self, call: ToolCall) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, call: ToolCall) -> ToolResult:
        try:
            before_path = float(self._executor.path_length)
        except Exception:
            before_path = 0.0
        try:
            evidence = self.invoke(call)
        except Exception as exc:  # noqa: BLE001 - intentional wrap
            return _error_result(call, exc)
        try:
            after_path = float(self._executor.path_length)
        except Exception:
            after_path = before_path
        return _evidence_to_result(
            call,
            evidence,
            after_path,
            terminal=self.terminal,
            path_delta=max(0.0, after_path - before_path),
        )


class ExplorePanoramaTool(_ExecutorNavigationTool):
    name: ClassVar[str] = "explore_panorama"
    arg_schema: ClassVar[str] = "no arguments (observe 8-view panorama in place)"

    def invoke(self, call: ToolCall) -> Any:
        return self._executor.explore_panorama(
            call.arguments.get("config", None)
        )


class NavigateToObjectTool(_ExecutorNavigationTool):
    name: ClassVar[str] = "navigate_to_object"
    arg_schema: ClassVar[str] = "required: object_name (str). optional: view_idx (int)"

    def invoke(self, call: ToolCall) -> Any:
        object_name = call.arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name:
            raise ValueError("navigate_to_object requires 'object_name'")
        view_idx = call.arguments.get("view_idx", None)
        return self._executor.navigate_to_object(object_name, view_idx)


class ExploreSeedTool(_ExecutorNavigationTool):
    name: ClassVar[str] = "explore_seed"
    arg_schema: ClassVar[str] = "required: seed_id (str) — pick from available_targets"

    def invoke(self, call: ToolCall) -> Any:
        seed_id = call.arguments.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id:
            raise ValueError("explore_seed requires 'seed_id'")
        return self._executor.explore_seed(seed_id)


class ExploreFrontierTool(_ExecutorNavigationTool):
    name: ClassVar[str] = "explore_frontier"
    arg_schema: ClassVar[str] = "required: frontier_id (str) — pick from available_targets"

    def invoke(self, call: ToolCall) -> Any:
        frontier_id = call.arguments.get("frontier_id")
        if not isinstance(frontier_id, str) or not frontier_id:
            raise ValueError("explore_frontier requires 'frontier_id'")
        return self._executor.explore_frontier(frontier_id)


def build_real_tool_registry(executor: _ExecutorLike) -> ToolRegistry:
    """Return a production :class:`ToolRegistry` backed by ``executor``.

    Wraps the four ``Executor`` navigation methods and reuses
    :class:`SubmitAnswerTool` (terminal, executor-independent). Does not
    register ``fork_subagent`` or ``pixel_navigate``. Navigation tool errors
    are caught and surfaced as ``ToolResult(ok=False)``; ``submit_answer``
    keeps its existing validation behavior.
    """
    registry = ToolRegistry()
    registry.register(ExplorePanoramaTool(executor))
    registry.register(NavigateToObjectTool(executor))
    registry.register(ExploreSeedTool(executor))
    registry.register(ExploreFrontierTool(executor))
    registry.register(SubmitAnswerTool())
    return registry
