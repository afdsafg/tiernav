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
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    RunSpec,
)
from .events import make_event
from .graph import RuntimeServices, build_runtime_graph
from .memory import MemoryService
from .policy import WorkflowPolicy
from .recorder import EpisodeRecorder
from .tools import ToolRegistry


class RuntimeEntrypoint:
    """Deterministic entrypoint that runs one episode through the runtime graph."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.graph = build_runtime_graph()

    @classmethod
    def with_fake_services(cls, planner: Any) -> "RuntimeEntrypoint":
        """Build an entrypoint backed by stable default fake services.

        No external services are contacted: tools are the noop/submit defaults,
        memory is in-memory, and the policy is the pure WorkflowPolicy.
        """
        services = RuntimeServices(
            planner=planner,
            tools=ToolRegistry.with_stable_defaults(),
            memory=MemoryService(),
            policy=WorkflowPolicy(),
        )
        return cls(services)

    def run(self, spec: RunSpec, request: EpisodeRequest) -> EpisodeResult:
        event_log_path = Path(spec.output_dir) / request.episode_id / "events.jsonl"
        recorder = EpisodeRecorder(event_log_path)

        recorder.append(
            make_event(
                episode_id=request.episode_id,
                event_type="episode_started",
                sequence=1,
                payload={"request": request.model_dump(mode="json")},
            )
        )

        final = self.graph.invoke(
            {
                "spec": spec.model_dump(mode="json"),
                "request": request.model_dump(mode="json"),
            },
            config={"configurable": {"services": self.services}},
        )
        state = EpisodeState.model_validate(final["state"])

        # EpisodeEndedPayload (replay.py) is extra=forbid and accepts exactly
        # success/answer/round_index/step_index, so the payload must carry only
        # those keys for the log to remain replayable.
        recorder.append(
            make_event(
                episode_id=request.episode_id,
                event_type="episode_ended",
                sequence=2,
                payload={
                    "success": state.success,
                    "answer": state.answer,
                    "round_index": state.round_index,
                    "step_index": state.step_index,
                },
            )
        )

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
            path_length=float(state.step_index),
            failure_type=state.failure_type,
            event_log_path=str(event_log_path),
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
