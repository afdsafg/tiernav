"""LangGraph runtime skeleton for the TierNav runtime.

Builds a deterministic :class:`StateGraph` over :class:`RuntimeGraphState`
backed by injectable services (planner, tools, memory, policy, context).
No external services are called; tests drive the graph with a scripted
planner and the stable default tool registry.

Node functions are closures over the :class:`RuntimeServices` bundle so the
service container never travels through graph state (it is not serializable
and has no place in a checkpoint). The declared :class:`RuntimeGraphState`
keys are the serializable contract: ``spec``/``request``/``state``/``policy``/
``prompt``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .context import ContextCompiler
from .contracts import (
    EpisodeRequest,
    EpisodeState,
    PlannerDecision,
    ToolCall,
    ToolResult,
    RunSpec,
)
from .memory import MemoryService
from .policy import WorkflowPolicy
from .tools import ToolRegistry


# --- Graph state -----------------------------------------------------------


class RuntimeGraphState(TypedDict):
    """Serializable state passed between LangGraph nodes.

    The service bundle is injected via closure in :func:`build_runtime_graph`
    and is intentionally not a state key: it is not serializable and should
    never be checkpointed.
    """

    spec: RunSpec
    request: EpisodeRequest
    state: Optional[EpisodeState]
    policy: WorkflowPolicy
    prompt: str


# --- Services container ----------------------------------------------------


@dataclass
class RuntimeServices:
    """Injectable service bundle consumed by runtime graph nodes.

    ``context`` defaults to a fresh :class:`ContextCompiler` in
    :meth:`__post_init__` so callers can omit it for the common case.
    """

    planner: object
    tools: ToolRegistry
    memory: MemoryService
    policy: WorkflowPolicy
    context: Optional[ContextCompiler] = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = ContextCompiler()


# --- Node factory ----------------------------------------------------------


def _build_nodes(services: RuntimeServices) -> dict[str, object]:
    """Create the node callables, closing over ``services``."""

    def bootstrap_node(state: RuntimeGraphState) -> dict:
        request: EpisodeRequest = state["request"]
        episode = EpisodeState(
            episode_id=request.episode_id,
            scene_id=request.scene_id,
            task_name=request.task_name,
            task_mode=request.task_mode,
            prompt=request.prompt,
            pose=dict(request.initial_pose),
        )
        return {"state": episode}

    def compile_context_node(state: RuntimeGraphState) -> dict:
        spec: RunSpec = state["spec"]
        episode: EpisodeState = state["state"]  # type: ignore[assignment]

        if spec.ablation.active_memory_query:
            pack = services.memory.query(episode.prompt)
            # Only retain a pack that carries real content; an empty summary
            # means memory had nothing to contribute this round.
            episode.memory_pack = pack if pack.summary else None

        sections = services.context.compile(
            episode,
            action_schema=services.tools.action_schema_text(),
            include_memory=spec.ablation.spatial_memory,
        )
        episode.context_sections = sections
        episode.prompt = services.context.render_prompt(sections)
        return {"state": episode, "prompt": episode.prompt}

    def plan_node(state: RuntimeGraphState) -> dict:
        episode: EpisodeState = state["state"]  # type: ignore[assignment]

        raw = services.planner.decide(episode.prompt)
        decision = (
            raw
            if isinstance(raw, PlannerDecision)
            else PlannerDecision.model_validate(raw)
        )
        episode.current_decision = decision
        episode.round_index += 1
        return {"state": episode}

    def policy_node(state: RuntimeGraphState) -> dict:
        spec: RunSpec = state["spec"]
        episode: EpisodeState = state["state"]  # type: ignore[assignment]

        decision = services.policy.decide(spec, episode)
        # Surface the fallback reason as failure_type for downstream nodes.
        if decision.route == "fallback":
            episode.failure_type = decision.reason
        return {"state": episode}

    def execute_tool_node(state: RuntimeGraphState) -> dict:
        episode: EpisodeState = state["state"]  # type: ignore[assignment]
        decision: PlannerDecision = episode.current_decision  # type: ignore[assignment]

        call = ToolCall(
            call_id=f"{episode.episode_id}-r{episode.round_index}-s{episode.step_index}",
            action_type=decision.action_type,
            arguments=dict(decision.arguments),
        )
        result: ToolResult = services.tools.dispatch(call)

        episode.last_observation = result.observation
        services.memory.update_from_observation(
            result.observation,
            action_type=result.action_type,
            round_index=episode.round_index,
        )

        if result.terminal:
            episode.terminal = True
            if result.ok:
                episode.success = True
                episode.answer = str(decision.arguments.get("answer", "") or "")
            else:
                episode.success = False
                episode.failure_type = episode.failure_type or result.error
            return {"state": episode}

        episode.step_index += 1
        return {"state": episode}

    def fallback_node(state: RuntimeGraphState) -> dict:
        spec: RunSpec = state["spec"]
        episode: EpisodeState = state["state"]  # type: ignore[assignment]

        decision = services.policy.decide(spec, episode)
        episode.terminal = True
        episode.success = False
        episode.answer = "unanswerable"
        episode.failure_type = decision.reason
        return {"state": episode}

    def recover_stall_node(state: RuntimeGraphState) -> dict:
        episode: EpisodeState = state["state"]  # type: ignore[assignment]
        episode.failure_type = ""
        return {"state": episode}

    def finalize_node(state: RuntimeGraphState) -> dict:
        episode: EpisodeState = state["state"]  # type: ignore[assignment]

        # fallback_node may have already finalized the episode (terminal +
        # answer set). Respect that and do not clobber with a submit_answer
        # read that has no answer argument.
        if episode.terminal:
            return {"state": episode}

        decision: PlannerDecision = episode.current_decision  # type: ignore[assignment]
        answer = str(decision.arguments.get("answer", "") or "")
        episode.terminal = True
        episode.success = bool(answer)
        episode.answer = answer
        return {"state": episode}

    return {
        "bootstrap": bootstrap_node,
        "compile_context": compile_context_node,
        "plan": plan_node,
        "policy": policy_node,
        "execute_tool": execute_tool_node,
        "fallback": fallback_node,
        "recover_stall": recover_stall_node,
        "finalize": finalize_node,
    }


# --- Graph builder ---------------------------------------------------------


def build_runtime_graph(services: RuntimeServices):
    """Compile and return the LangGraph runtime app.

    Edge topology (matches the Task 7 plan):

        START -> bootstrap -> compile_context -> plan -> policy
        policy --conditional--> execute_tool | finalize | fallback | recover_stall
        execute_tool --conditional--> END (terminal) | compile_context (next round)
        recover_stall -> compile_context
        fallback -> finalize
        finalize -> END
    """
    nodes = _build_nodes(services)
    graph = StateGraph(RuntimeGraphState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "compile_context")
    graph.add_edge("compile_context", "plan")
    graph.add_edge("plan", "policy")

    graph.add_conditional_edges(
        "policy",
        _route_after_policy_factory(services),
        {
            "execute_tool": "execute_tool",
            "finalize": "finalize",
            "fallback": "fallback",
            "recover_stall": "recover_stall",
        },
    )

    graph.add_conditional_edges(
        "execute_tool",
        _route_after_execute_factory(),
        {
            END: END,
            "compile_context": "compile_context",
        },
    )

    graph.add_edge("recover_stall", "compile_context")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Router factories kept module-level so they are picklable and testable.
# They close over ``services`` only where needed.


def _route_after_policy_factory(services: RuntimeServices):
    def route_after_policy(state: RuntimeGraphState) -> str:
        spec: RunSpec = state["spec"]
        episode: EpisodeState = state["state"]  # type: ignore[assignment]
        return services.policy.decide(spec, episode).route

    return route_after_policy


def _route_after_execute_factory():
    def route_after_execute(state: RuntimeGraphState) -> str:
        episode: EpisodeState = state["state"]  # type: ignore[assignment]
        if episode.terminal:
            return END
        # Re-enter the round loop: recompile context, then replan.
        return "compile_context"

    return route_after_execute
