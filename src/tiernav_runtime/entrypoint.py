"""Deterministic runtime entrypoint and legacy-compatible return mapping.

The entrypoint runs one episode against the LangGraph runtime with injectable
services, writes an append-only event log, and maps the resulting
:class:`EpisodeResult` back to the legacy runner dict shape.

This is a deterministic dev/replay path: :meth:`RuntimeEntrypoint.with_fake_services`
wires stable default tools, an in-memory memory service, and the default
workflow policy. It must not be wired into production runners; Habitat-backed
services are mapped separately once available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import (
    BenchmarkRule,
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    RunSpec,
)
from .events import make_event
from .graph import RuntimeServices, build_runtime_graph
from .memory import MemoryService, MemorySession
from .policy import WorkflowPolicy
from .recorder import EpisodeRecorder, PromptAuditRecorder
from .success import SuccessEvaluator
from .tools import ToolRegistry, build_real_tool_registry


class RuntimeEntrypoint:
    """Deterministic entrypoint that runs one episode through the runtime graph."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.graph = build_runtime_graph()

    @classmethod
    def with_fake_services(cls, planner: Any) -> "RuntimeEntrypoint":
        """Build an entrypoint backed by stable default fake services.

        No external services are contacted: tools are the noop/submit defaults,
        memory is in-memory, and the policy is the default WorkflowPolicy.
        """
        services = RuntimeServices(
            planner=planner,
            tools=ToolRegistry.with_stable_defaults(),
            memory=MemoryService(),
            policy=WorkflowPolicy(),
        )
        return cls(services)

    @classmethod
    def with_environment_services(
        cls,
        planner: Any,
        environment: Any,
        memory: MemoryService | None = None,
        policy: WorkflowPolicy | None = None,
    ) -> "RuntimeEntrypoint":
        """Build an entrypoint backed by a RuntimeEnvironmentService.

        The environment service owns the Habitat scene, TSDFPlanner, models,
        and pose/path_length state. Tools remain the stable defaults for now;
        Task 4 wires Habitat-backed tools and Task 7 rewires the graph to call
        the environment service. This factory only makes the service available
        on :class:`RuntimeServices` so production runners can construct it.
        """
        services = RuntimeServices(
            planner=planner,
            tools=ToolRegistry.with_stable_defaults(),
            memory=memory if memory is not None else MemoryService(),
            policy=policy if policy is not None else WorkflowPolicy(),
            environment=environment,
        )
        return cls(services)

    @classmethod
    def with_real_services(
        cls,
        planner: Any,
        environment: Any,
        rule: BenchmarkRule,
        executor: Any,
        *,
        memory_scope_adapter: MemorySession | None = None,
        policy: WorkflowPolicy | None = None,
        task_mode: str = "",
    ) -> "RuntimeEntrypoint":
        """Build an entrypoint backed by real production services.

        Wires the full service stack: real executor-backed tool registry,
        SuccessEvaluator, environment service, and optional MemorySession.
        The graph consumes these services through RuntimeServices.
        """
        services = RuntimeServices(
            planner=planner,
            tools=build_real_tool_registry(executor, task_mode=task_mode),
            memory=memory_scope_adapter if memory_scope_adapter is not None else MemoryService(),
            policy=policy if policy is not None else WorkflowPolicy(),
            environment=environment,
            memory_session=memory_scope_adapter,
            success_evaluator=SuccessEvaluator(rule),
            rule=rule,
        )
        return cls(services)

    def run(self, spec: RunSpec, request: EpisodeRequest) -> EpisodeResult:
        """Run one episode through the runtime graph and write its event log.

        The event log is append-only: if a log already exists for this episode
        (e.g. a prior run with the same episode_id and output_dir), this raises
        ``FileExistsError`` rather than overwriting or appending to it, so a
        re-run cannot pollute a previously recorded episode.

        Note: ``graph.invoke`` runs after ``episode_started`` is appended. If it
        raises, the log is left with only ``episode_started``; callers should
        treat a partial log as a failed run rather than replay it as terminal.
        """
        self._ensure_memory_session_started(request)
        event_log_path = Path(spec.output_dir) / request.episode_id / "events.jsonl"
        if event_log_path.exists():
            raise FileExistsError(
                f"event log already exists for episode_id={request.episode_id!r} "
                f"at {event_log_path}; refusing to overwrite append-only log"
            )
        recorder = EpisodeRecorder(event_log_path)
        self.services.recorder = recorder
        self.services.event_seq[0] = 2
        if spec.output_dir:
            self.services.prompt_audit = PromptAuditRecorder(spec.output_dir)

        # Phase 2: wire SceneMemoryStore for cross-episode recall + sediment.
        # Per-scene: created here (not at construction) because scene_id is
        # request-scoped. Cleared in finally so services stay reusable.
        if spec.output_dir:
            from .scene_memory import SceneMemoryStore

            self.services.scene_memory_store = SceneMemoryStore(
                scene_id=request.scene_id,
                output_dir=spec.output_dir,
            )
            # Register query_scene_memory lazily: build_real_tool_registry ran
            # at construction before scene_id was known, so the store was not
            # wired then. Register now if not already present.
            if "query_scene_memory" not in self.services.tools.names():
                from .tools import QuerySceneMemoryTool

                self.services.tools.register(
                    QuerySceneMemoryTool(
                        self.services.scene_memory_store, self.services.planner
                    )
                )
        else:
            self.services.scene_memory_store = None

        recorder.append(
            make_event(
                episode_id=request.episode_id,
                event_type="episode_started",
                sequence=1,
                payload={"request": request.model_dump(mode="json")},
            )
        )

        try:
            final = self.graph.invoke(
                {
                    "spec": spec.model_dump(mode="json"),
                    "request": request.model_dump(mode="json"),
                },
                config={
                    "configurable": {"services": self.services},
                    # Each round traverses ~4 nodes (compile_context, plan, policy,
                    # execute_tool). Default LangGraph recursion_limit=25 caps at
                    # ~6 rounds; scale to spec.max_rounds * 8 + 10 headroom.
                    "recursion_limit": max(25, int(spec.max_rounds) * 8 + 10),
                },
            )
            state = EpisodeState.model_validate(final["state"])

            # EpisodeEndedPayload (replay.py) is extra=forbid and accepts exactly
            # success/answer/round_index/step_index, so the payload must carry only
            # those keys for the log to remain replayable.
            recorder.append(
                make_event(
                    episode_id=request.episode_id,
                    event_type="episode_ended",
                    sequence=self.services.event_seq[0] + 1,
                    payload={
                        "success": state.success,
                        "answer": state.answer,
                        "round_index": state.round_index,
                        "step_index": state.step_index,
                    },
                )
            )
        finally:
            # Clear recorder so the services object is reusable across episodes
            # even when graph.invoke raises (partial log left to caller).
            # scene_memory_store is per-scene; clear so next episode gets fresh.
            self.services.recorder = None
            self.services.scene_memory_store = None

        return EpisodeResult(
            schema_version=state.schema_version,
            episode_id=state.episode_id,
            scene_id=state.scene_id,
            task_name=state.task_name,
            task_mode=state.task_mode,
            success=state.success,
            answer=state.answer,
            steps_taken=state.step_index,
            rounds_used=state.round_index,
            path_length=self._resolve_path_length(state),
            failure_type=state.failure_type,
            event_log_path=str(event_log_path),
            distance_to_goal=state.distance_to_goal,
            submit_was_explicit=state.submitted_explicitly,
        )

    def _resolve_path_length(self, state: EpisodeState) -> float:
        """Prefer the environment service's real path_length; fall back to 0.0."""
        env = self.services.environment
        if env is not None and hasattr(env, "path_length"):
            try:
                return float(env.path_length)
            except (TypeError, ValueError):
                pass
        return 0.0

    def _ensure_memory_session_started(self, request: EpisodeRequest) -> None:
        session = self.services.memory_session
        if session is None:
            return
        try:
            session.current_memory
            return
        except RuntimeError:
            pass
        subtask_index = request.goal_metadata.get("subtask_index")
        question_id = request.episode_id if request.task_name == "aeqa" else None
        session.start_session(
            episode_id=request.episode_id,
            question_id=question_id,
            subtask_index=(
                int(subtask_index) if isinstance(subtask_index, int) else None
            ),
        )


def episode_result_to_legacy_dict(
    result: EpisodeResult, question: str = ""
) -> dict[str, Any]:
    """Map an :class:`EpisodeResult` to the legacy runner dict shape.

    Returns the keys the legacy AEQA/GOATBench runners read from a result dict,
    including the snapshot counts (default 0 — the fake runtime produces none)
    and ``event_log_path`` for downstream replay tooling.
    """
    return {
        "scene_id": result.scene_id,
        "question_id": result.episode_id,
        "question": question,
        "answer": result.answer,
        "success": result.success,
        "steps_taken": result.steps_taken,
        "rounds_used": result.rounds_used,
        "path_length": result.path_length,
        "n_filtered_snapshots": 0,
        "n_total_snapshots": 0,
        "error": result.error,
        "event_log_path": result.event_log_path,
        "failure_type": result.failure_type,
    }
