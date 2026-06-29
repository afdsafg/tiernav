"""LangGraph runtime skeleton for the TierNav runtime.

Builds a deterministic :class:`StateGraph` over :class:`RuntimeGraphState`
backed by injectable services (planner, tools, memory, policy, context).
No external services are called; tests drive the graph with a scripted
planner and the stable default tool registry.

Services are injected through the LangGraph invoke config:
``config={"configurable": {"services": services}}``. The service bundle is
not serializable and never travels through graph state; node functions read
it via the :func:`_services` helper.

The declared :class:`RuntimeGraphState` keys are the serializable JSON
contract: ``spec``/``request``/``state``/``policy``/``prompt``. Each node
round-trips pydantic models via ``model_validate`` /
``model_dump(mode="json")`` so the graph state is always a plain JSON dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from .context import ContextCompiler
from .contracts import (
    EpisodeRequest,
    EpisodeState,
    PlannerDecision,
    RunSpec,
    ToolCall,
    ToolResult,
)
from .memory import MemoryService, MemorySession
from .policy import PolicyDecision, WorkflowPolicy
from .tools import ToolRegistry


# --- Graph state -----------------------------------------------------------


class RuntimeGraphState(TypedDict, total=False):
    """Serializable JSON state passed between LangGraph nodes.

    All values are plain JSON (dicts/str). Pydantic models are reconstructed
    inside nodes via ``model_validate`` and written back via
    ``model_dump(mode="json")``. ``total=False`` lets partial node returns
    merge into state without requiring every key on every return.
    """

    spec: dict[str, Any]
    request: dict[str, Any]
    state: dict[str, Any]
    policy: dict[str, Any]
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
    context: ContextCompiler | None = None
    environment: object | None = None
    # ponytail: Task 8 (adapters) wires a MemorySession here to give the graph
    # benchmark-correct memory lifetime (AEQA per-question reset vs GOATBench
    # cross-subtask persistence). Until then, graph nodes keep reading/writing
    # `memory` directly, so existing behavior is unchanged.
    memory_session: MemorySession | None = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = ContextCompiler()


def _services(config: RunnableConfig) -> RuntimeServices:
    """Extract the :class:`RuntimeServices` bundle from a LangGraph config."""
    configurable = (config or {}).get("configurable") or {}
    services = configurable.get("services")
    if services is None:
        raise KeyError(
            "RuntimeServices must be injected via "
            "config['configurable']['services']"
        )
    return services


# --- Nodes -----------------------------------------------------------------


def bootstrap_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    request = EpisodeRequest.model_validate(graph_state["request"])
    episode = EpisodeState(
        episode_id=request.episode_id,
        scene_id=request.scene_id,
        task_name=request.task_name,
        task_mode=request.task_mode,
        prompt=request.prompt,
        pose=dict(request.initial_pose),
    )
    return {"state": episode.model_dump(mode="json")}


def compile_context_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    services = _services(config)
    spec = RunSpec.model_validate(graph_state["spec"])
    episode = EpisodeState.model_validate(graph_state["state"])

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
    return {
        "state": episode.model_dump(mode="json"),
        "prompt": episode.prompt,
    }


def plan_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    services = _services(config)
    episode = EpisodeState.model_validate(graph_state["state"])

    raw = services.planner.decide(episode.prompt)
    decision = (
        raw
        if isinstance(raw, PlannerDecision)
        else PlannerDecision.model_validate(raw)
    )
    episode.current_decision = decision
    episode.round_index += 1
    return {"state": episode.model_dump(mode="json")}


def policy_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    services = _services(config)
    spec = RunSpec.model_validate(graph_state["spec"])
    episode = EpisodeState.model_validate(graph_state["state"])

    decision = services.policy.decide(spec, episode)
    # Surface the fallback reason as failure_type for downstream nodes.
    if decision.route == "fallback":
        episode.failure_type = decision.reason
    return {
        "state": episode.model_dump(mode="json"),
        "policy": decision.model_dump(mode="json"),
    }


def execute_tool_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    services = _services(config)
    episode = EpisodeState.model_validate(graph_state["state"])
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

    episode.step_index += 1
    return {"state": episode.model_dump(mode="json")}


def fallback_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    episode = EpisodeState.model_validate(graph_state["state"])
    policy = PolicyDecision.model_validate(graph_state["policy"])

    episode.terminal = True
    episode.success = False
    episode.answer = "unanswerable"
    episode.failure_type = policy.reason
    return {"state": episode.model_dump(mode="json")}


def recover_stall_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    episode = EpisodeState.model_validate(graph_state["state"])
    episode.failure_type = ""
    return {"state": episode.model_dump(mode="json")}


def finalize_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    episode = EpisodeState.model_validate(graph_state["state"])

    # fallback_node (or a terminal execute_tool result) may have already
    # finalized the episode. Respect that and do not clobber answer /
    # failure_type with a submit_answer read that has no answer argument.
    if episode.terminal:
        return {"state": episode.model_dump(mode="json")}

    decision: PlannerDecision = episode.current_decision  # type: ignore[assignment]
    answer = str(decision.arguments.get("answer", "") or "")
    episode.terminal = True
    episode.success = bool(answer)
    episode.answer = answer
    return {"state": episode.model_dump(mode="json")}


# --- Routers ---------------------------------------------------------------


def route_after_policy(graph_state: RuntimeGraphState) -> str:
    decision = PolicyDecision.model_validate(graph_state["policy"])
    return decision.route


def route_after_execute(graph_state: RuntimeGraphState) -> str:
    episode = EpisodeState.model_validate(graph_state["state"])
    return "finalize" if episode.terminal else "compile_context"


# --- Graph builder ---------------------------------------------------------


def build_runtime_graph():
    """Compile and return the LangGraph runtime app.

    Edge topology (matches the Task 7 plan):

        START -> bootstrap -> compile_context -> plan -> policy
        policy --conditional--> execute_tool | finalize | fallback | recover_stall
        execute_tool --conditional--> finalize (terminal) | compile_context (next round)
        recover_stall -> compile_context
        fallback -> finalize
        finalize -> END
    """
    graph = StateGraph(RuntimeGraphState)

    graph.add_node("bootstrap", bootstrap_node)
    graph.add_node("compile_context", compile_context_node)
    graph.add_node("plan", plan_node)
    graph.add_node("policy", policy_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("recover_stall", recover_stall_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "compile_context")
    graph.add_edge("compile_context", "plan")
    graph.add_edge("plan", "policy")

    graph.add_conditional_edges(
        "policy",
        route_after_policy,
        {
            "execute_tool": "execute_tool",
            "finalize": "finalize",
            "fallback": "fallback",
            "recover_stall": "recover_stall",
        },
    )

    graph.add_conditional_edges(
        "execute_tool",
        route_after_execute,
        {
            "compile_context": "compile_context",
            "finalize": "finalize",
        },
    )

    graph.add_edge("recover_stall", "compile_context")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
