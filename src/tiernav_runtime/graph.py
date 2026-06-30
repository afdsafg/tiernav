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
from .events import make_event
from .memory import MemoryService, MemorySession
from .policy import PolicyDecision, WorkflowPolicy
from .recorder import EpisodeRecorder
from .success import SuccessEvaluator
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
    # Task 7: wire success evaluation through the graph. When None, the graph
    # falls back to legacy AEQA logic (answer non-empty = success).
    success_evaluator: SuccessEvaluator | None = None
    # Active BenchmarkRule, set by with_real_services, read by plan_node
    # for planner_retries. None for fake-services path.
    rule: Any = None
    # EpisodeRecorder wired by the entrypoint so graph nodes can append
    # design-spec events. None on the fake/dev path and between episodes.
    recorder: EpisodeRecorder | None = None
    # ponytail: `subtask_started` (design-spec event 11) is GOATBench-only and
    # emitted by the runner at subtask boundaries, not by a graph node — the
    # graph has no subtask concept. Add a runner-side emit if GOATBench needs
    # it in the event log.
    # Monotonic intra-episode event sequence counter. episode_started=1 is
    # appended by the entrypoint; _emit increments before appending, so the
    # first intra-episode event gets sequence 3 (2 is the pre-increment seed).
    # Boxed in a list so the dataclass default stays mutable-safe.
    event_seq: list[int] = None  # set in __post_init__

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = ContextCompiler()
        if self.event_seq is None:
            self.event_seq = [2]


def _emit(
    services: "RuntimeServices",
    episode_id: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    """Append a design-spec event if a recorder is wired."""
    if services.recorder is None:
        return
    services.event_seq[0] += 1
    services.recorder.append(
        make_event(episode_id, event_type, services.event_seq[0], payload or {})
    )


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
        env=services.environment,
    )
    episode.context_sections = sections
    episode.prompt = services.context.render_prompt(sections)
    _emit(services, episode.episode_id, "context_compiled", {
        "sections": [s.model_dump(mode="json") for s in sections],
        "memory_query_used": episode.memory_pack is not None,
    })
    if episode.memory_pack is not None:
        _emit(services, episode.episode_id, "memory_query", {
            "summary": episode.memory_pack.summary,
        })
    return {
        "state": episode.model_dump(mode="json"),
        "prompt": episode.prompt,
    }


def plan_node(
    graph_state: RuntimeGraphState, config: RunnableConfig
) -> RuntimeGraphState:
    services = _services(config)
    episode = EpisodeState.model_validate(graph_state["state"])

    _emit(services, episode.episode_id, "planner_called", {
        "prompt": episode.prompt,
        "round_index": episode.round_index + 1,
    })
    rule = getattr(services, "rule", None)
    retries = getattr(rule, "planner_retries", 0) if rule is not None else 0
    raw = services.planner.decide(episode.prompt, retries=retries)
    decision = (
        raw
        if isinstance(raw, PlannerDecision)
        else PlannerDecision.model_validate(raw)
    )
    episode.current_decision = decision
    episode.round_index += 1
    _emit(services, episode.episode_id, "planner_decision", {
        "action_type": decision.action_type,
        "reasoning": decision.reasoning,
        "arguments": decision.arguments,
        "round_index": episode.round_index,
    })
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
    _emit(services, episode.episode_id, "tool_called", {
        "call_id": call.call_id, "action_type": call.action_type,
        "arguments": call.arguments, "step_index": episode.step_index,
    })
    result: ToolResult = services.tools.dispatch(call)

    # Sync executor pose back to the environment service so distance-to-goal
    # stays fresh across steps (critical for GOATBench 1m success check).
    env = services.environment
    executor = getattr(env, "executor", None) if env is not None else None
    if executor is not None and hasattr(executor, "_pts") and executor._pts is not None:
        pts = executor._pts
        angle = getattr(executor, "_angle", 0.0) or 0.0
        # Habitat pts is 3D [x, y, z] where y is up; preserve all 3 so the
        # pose dict round-trips through _reset_session with correct dimensionality.
        env._current_pose = {
            "x": float(pts[0]) if len(pts) > 0 else 0.0,
            "y": float(pts[1]) if len(pts) > 1 else 0.0,
            "z": float(pts[2]) if len(pts) > 2 else 0.0,
            "theta": float(angle),
        }
        env._path_length = float(getattr(executor, "_path_length", 0.0) or 0.0)

    episode.last_observation = result.observation
    services.memory.update_from_observation(
        result.observation,
        action_type=result.action_type,
        round_index=episode.round_index,
    )
    _emit(services, episode.episode_id, "tool_result", {
        "observation": result.observation.model_dump(mode="json"),
        "ok": result.ok, "terminal": result.terminal,
        "error": result.error,
        "step_index": episode.step_index,
    })
    _emit(services, episode.episode_id, "memory_updated", {
        "action_type": result.action_type,
        "round_index": episode.round_index,
    })

    if result.terminal:
        episode.terminal = True
        if result.ok:
            episode.success = True
            episode.answer = str(decision.arguments.get("answer", "") or "")
        else:
            episode.success = False
            episode.failure_type = episode.failure_type or result.error

    # Track whether this step was an explicit submit. The finalize_node
    # uses this to inform the SuccessEvaluator (GOATBench requires
    # explicit-planner-stop).
    episode.submitted_explicitly = (
        result.terminal
        and result.ok
        and decision.action_type == "submit_answer"
    )

    # Compute agent-to-goal distance when the environment service is
    # available (GOATBench episodes). Skip when env is None (fake path).
    env = services.environment
    if env is not None and hasattr(env, "distance_to_goal"):
        dist = env.distance_to_goal()
        if dist is not None:
            episode.distance_to_goal = dist

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
    episode.submitted_explicitly = False
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
    services = _services(config)
    episode = EpisodeState.model_validate(graph_state["state"])

    # fallback_node (or a terminal execute_tool result) may have already
    # finalized the episode. Respect that and do not clobber answer /
    # failure_type with a submit_answer read that has no answer argument.
    if episode.terminal:
        return {"state": episode.model_dump(mode="json")}

    # --- Evaluator path (real services) ---
    # When a SuccessEvaluator is injected, use it for both AEQA and
    # GOATBench success verdicts. Distance is accumulated by
    # execute_tool_node (updated each round for navigation actions);
    # submitted_explicitly derives from the planner decision here.
    evaluator = services.success_evaluator
    if evaluator is not None:
        decision: PlannerDecision = episode.current_decision  # type: ignore[assignment]
        submitted_explicitly = (
            decision is not None
            and decision.action_type == "submit_answer"
        )
        answer = str(decision.arguments.get("answer", "") or "")
        if submitted_explicitly:
            episode.answer = answer

        # Refresh distance from environment when available (GOATBench).
        env = services.environment
        if env is not None and hasattr(env, "distance_to_goal"):
            dist = env.distance_to_goal()
            if dist is not None:
                episode.distance_to_goal = dist

        verdict = evaluator.evaluate(
            episode.task_mode,
            submitted_explicitly=submitted_explicitly,
            answer=episode.answer,
            distance_to_goal=episode.distance_to_goal,
        )
        episode.terminal = True
        episode.success = verdict.success
        episode.failure_type = episode.failure_type or verdict.reason
        episode.submitted_explicitly = submitted_explicitly
        _emit(services, episode.episode_id, "success_evaluated", {
            "success": episode.success, "answer": episode.answer,
            "submitted_explicitly": submitted_explicitly,
            "distance_to_goal": episode.distance_to_goal,
            "failure_type": episode.failure_type,
        })
        return {"state": episode.model_dump(mode="json")}

    # --- Legacy AEQA path (fake services, no evaluator) ---
    decision: PlannerDecision = episode.current_decision  # type: ignore[assignment]
    answer = str(decision.arguments.get("answer", "") or "")
    episode.terminal = True
    episode.success = bool(answer)
    episode.answer = answer
    _emit(services, episode.episode_id, "success_evaluated", {
        "success": episode.success, "answer": episode.answer,
        "submitted_explicitly": False,
        "distance_to_goal": episode.distance_to_goal,
        "failure_type": episode.failure_type,
    })
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
