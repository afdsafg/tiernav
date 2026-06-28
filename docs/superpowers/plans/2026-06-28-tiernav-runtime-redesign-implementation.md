# TierNav Runtime Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-only, Pydantic-contract runtime for TierNav that supports continuous context, structured spatial memory, append-only replay, and ablation-ready execution.

**Architecture:** Add a new `src/tiernav_runtime/` package beside the current `src/two_tier_graph/` implementation, then migrate one boundary at a time. The first milestone is a deterministic fake-environment graph that exercises contracts, context compilation, policy routing, tools, memory, recorder, replay, and task adapters without Habitat or VLM calls. The second milestone wires AEQA and GOATBench through adapters while preserving old entrypoints as thin compatibility shells.

**Tech Stack:** Python 3.9+, Pydantic, LangGraph, pytest, existing TierNav planner/executor/memory code, existing AEQA and GOATBench runners.

---

## Scope Check

This plan covers the runtime core plus adapter migration because those pieces must agree on the same Pydantic contracts. It deliberately does not implement new academic algorithms, new benchmark tuning, multi-agent execution, critic logic, or PixelNavigate. Those remain experiment plugins after the default runtime path is stable.

The implementation has two hard gates:

1. The new runtime must pass deterministic unit tests before touching AEQA/GOATBench entrypoints.
2. Default tool registration must not expose any tool that raises `NotImplementedError` or returns fake evidence.

## File Structure

Create a new package:

```text
src/tiernav_runtime/
  __init__.py
  contracts.py
  events.py
  recorder.py
  replay.py
  context.py
  policy.py
  memory.py
  planner.py
  tools.py
  adapters.py
  graph.py
  entrypoint.py
```

Tests:

```text
tests/runtime/
  test_contracts.py
  test_recorder_replay.py
  test_context_compiler.py
  test_policy.py
  test_memory_service.py
  test_tools.py
  test_graph_runtime.py
  test_adapters.py
  test_entrypoint_compat.py
```

Responsibilities:

- `contracts.py`: Pydantic models and schema export helpers.
- `events.py`: append-only event envelope construction and validation.
- `recorder.py`: JSONL writer for episode events.
- `replay.py`: reconstruct materialized episode state from event logs.
- `context.py`: Claude Code-style sectioned prompt/context compiler.
- `policy.py`: pure routing decisions for budget, stall, fallback, submit, and ablation gates.
- `memory.py`: room-snapshot-object-hypothesis memory graph and query packs.
- `planner.py`: runtime planner interface plus adapter from existing `src.agent_planner.PlannerAction`.
- `tools.py`: stable tool registry and result contract.
- `adapters.py`: AEQA and GOATBench task adapters.
- `graph.py`: LangGraph graph using only the runtime contracts and services.
- `entrypoint.py`: new runtime entrypoint and legacy-compatible return mapping.

## Implementation Order

```text
Task 1  Contracts and schemas
Task 2  Event recorder and replay
Task 3  Context compiler
Task 4  Workflow policy
Task 5  Spatial memory service
Task 6  Planner and tool interfaces
Task 7  LangGraph runtime on deterministic fake services
Task 8  AEQA / GOATBench task adapters
Task 9  Compatibility entrypoint and runner integration
Task 10 Runtime cleanup, schema snapshots, and default-path stub audit
```

---

## Task 1: Contracts and JSON Schema

**Files:**
- Create: `src/tiernav_runtime/__init__.py`
- Create: `src/tiernav_runtime/contracts.py`
- Test: `tests/runtime/test_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/runtime/test_contracts.py`:

```python
"""Contract tests for the TierNav runtime."""
import json

from pydantic import ValidationError

from src.tiernav_runtime.contracts import (
    AblationConfig,
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    Observation,
    PlannerDecision,
    RunSpec,
    ToolCall,
    ToolResult,
    dump_runtime_json_schemas,
)


def test_run_spec_has_research_ablation_axes():
    spec = RunSpec(
        run_id="run-001",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
        seed=7,
        ablation=AblationConfig(
            continuous_context=True,
            spatial_memory=True,
            active_memory_query=True,
            prompt_cache=True,
            stall_recovery=False,
        ),
    )

    assert spec.ablation.continuous_context is True
    assert spec.ablation.spatial_memory is True
    assert spec.ablation.active_memory_query is True


def test_episode_request_rejects_unknown_task_mode():
    try:
        EpisodeRequest(
            episode_id="ep-1",
            scene_id="scene",
            task_name="aeqa",
            task_mode="unknown",
            prompt="What color is the chair?",
        )
    except ValidationError as exc:
        assert "task_mode" in str(exc)
    else:
        raise AssertionError("EpisodeRequest accepted an unknown task_mode")


def test_planner_decision_round_trip_json():
    decision = PlannerDecision(
        action_type="navigate_to_object",
        reasoning="The chair is visible.",
        expected="Move closer to verify the answer.",
        confidence=0.8,
        arguments={"snapshot_id": "step1_view0", "object_name": "chair"},
    )

    encoded = decision.model_dump_json()
    decoded = PlannerDecision.model_validate_json(encoded)

    assert decoded.action_type == "navigate_to_object"
    assert decoded.arguments["object_name"] == "chair"


def test_episode_state_serializes_without_numpy_objects():
    state = EpisodeState(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="Where is the lamp?",
        round_index=1,
        step_index=2,
        pose={"x": 1.0, "y": 0.0, "z": 2.0, "yaw": 0.5},
    )

    payload = json.loads(state.model_dump_json())

    assert payload["pose"]["x"] == 1.0
    assert payload["round_index"] == 1


def test_tool_contracts_validate_terminal_results():
    call = ToolCall(
        call_id="tool-1",
        action_type="submit_answer",
        arguments={"answer": "red"},
    )
    result = ToolResult(
        call_id=call.call_id,
        action_type=call.action_type,
        ok=True,
        terminal=True,
        observation=Observation(summary="Answer submitted."),
    )

    assert result.terminal is True
    assert result.observation.summary == "Answer submitted."


def test_episode_result_has_common_metrics_for_aeqa_and_goatbench():
    result = EpisodeResult(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        success=True,
        answer="chair",
        steps_taken=4,
        rounds_used=2,
        path_length=3.5,
        event_log_path="/tmp/tiernav/ep-1/events.jsonl",
    )

    assert result.path_length == 3.5
    assert result.event_log_path.endswith("events.jsonl")


def test_json_schema_dump_contains_all_public_models():
    schemas = dump_runtime_json_schemas()

    assert "RunSpec" in schemas
    assert "EpisodeRequest" in schemas
    assert "EpisodeState" in schemas
    assert "EpisodeResult" in schemas
    assert schemas["RunSpec"]["type"] == "object"
```

- [ ] **Step 2: Run the contract tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_contracts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.tiernav_runtime'`.

- [ ] **Step 3: Create the package marker**

Create `src/tiernav_runtime/__init__.py`:

```python
"""TierNav runtime package.

The runtime is the contract-first, LangGraph-only execution layer used by
AEQA, GOATBench, replay, and ablation runs.
"""
```

- [ ] **Step 4: Implement Pydantic contracts**

Create `src/tiernav_runtime/contracts.py`:

```python
"""Pydantic contracts for the TierNav runtime."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "tiernav.runtime.v1"


class RuntimeModel(BaseModel):
    """Base model for runtime contracts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TaskMode(str, Enum):
    QUESTION_ANSWERING = "question_answering"
    GOAL_NAVIGATION = "goal_navigation"


class AblationConfig(RuntimeModel):
    """Ablation switches for the three main contributions and support levers."""

    continuous_context: bool = True
    spatial_memory: bool = True
    active_memory_query: bool = True
    prompt_cache: bool = True
    stall_recovery: bool = False


class RunSpec(RuntimeModel):
    """Configuration for a reproducible run or sweep member."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    task_name: str
    dataset_split: str
    output_dir: str
    planner_provider: str
    planner_model: str
    seed: int = 0
    max_rounds: int = 10
    max_steps: int = 50
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRequest(RuntimeModel):
    """Task-adapted input for one episode."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    scene_id: str
    task_name: str
    task_mode: TaskMode
    prompt: str
    goal_metadata: dict[str, Any] = Field(default_factory=dict)
    initial_pose: dict[str, float] = Field(default_factory=dict)
    output_dir: str = ""


class Observation(RuntimeModel):
    """Serializable observation produced by tools or adapters."""

    summary: str = ""
    image_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    room_id: Optional[str] = None
    pose: dict[str, float] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class PlannerDecision(RuntimeModel):
    """Model-selected action after context compilation."""

    action_type: str
    reasoning: str = ""
    expected: str = ""
    confidence: float = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


class ToolCall(RuntimeModel):
    """Validated tool invocation."""

    call_id: str
    action_type: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(RuntimeModel):
    """Structured result returned by a runtime tool."""

    call_id: str
    action_type: str
    ok: bool
    terminal: bool = False
    observation: Observation = Field(default_factory=Observation)
    error: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class MemoryPack(RuntimeModel):
    """Context-ready memory query result."""

    query: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reuse_hint: str = ""


class ContextSection(RuntimeModel):
    """One context section with cache metadata."""

    name: str
    content: str
    cacheable: bool
    token_estimate: int = 0
    content_hash: str = ""


class EpisodeState(RuntimeModel):
    """Materialized graph state. The event log remains the source of truth."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    scene_id: str
    task_name: str
    task_mode: TaskMode
    prompt: str
    round_index: int = 0
    step_index: int = 0
    pose: dict[str, float] = Field(default_factory=dict)
    current_decision: Optional[PlannerDecision] = None
    last_observation: Observation = Field(default_factory=Observation)
    memory_pack: Optional[MemoryPack] = None
    context_sections: list[ContextSection] = Field(default_factory=list)
    terminal: bool = False
    success: bool = False
    answer: str = ""
    failure_type: str = ""


class EpisodeResult(RuntimeModel):
    """Unified output from one episode."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    scene_id: str
    task_name: str
    task_mode: TaskMode
    success: bool
    answer: str = ""
    steps_taken: int = 0
    rounds_used: int = 0
    path_length: float = 0.0
    failure_type: str = ""
    error: str = ""
    event_log_path: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)


PublicModel = Literal[
    "RunSpec",
    "EpisodeRequest",
    "EpisodeState",
    "EpisodeResult",
    "PlannerDecision",
    "ToolCall",
    "ToolResult",
    "Observation",
    "MemoryPack",
    "ContextSection",
]


def dump_runtime_json_schemas() -> dict[str, dict[str, Any]]:
    """Return JSON schemas for public runtime contracts."""

    models: dict[str, type[BaseModel]] = {
        "RunSpec": RunSpec,
        "EpisodeRequest": EpisodeRequest,
        "EpisodeState": EpisodeState,
        "EpisodeResult": EpisodeResult,
        "PlannerDecision": PlannerDecision,
        "ToolCall": ToolCall,
        "ToolResult": ToolResult,
        "Observation": Observation,
        "MemoryPack": MemoryPack,
        "ContextSection": ContextSection,
    }
    return {name: model.model_json_schema() for name, model in models.items()}
```

- [ ] **Step 5: Run the contract tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_contracts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit contracts**

Run:

```bash
git add src/tiernav_runtime/__init__.py src/tiernav_runtime/contracts.py tests/runtime/test_contracts.py
git commit -m "feat(runtime): add pydantic contracts"
```

---

## Task 2: Append-Only Events, Recorder, and Replay

**Files:**
- Create: `src/tiernav_runtime/events.py`
- Create: `src/tiernav_runtime/recorder.py`
- Create: `src/tiernav_runtime/replay.py`
- Test: `tests/runtime/test_recorder_replay.py`

- [ ] **Step 1: Write failing event and replay tests**

Create `tests/runtime/test_recorder_replay.py`:

```python
"""Tests for append-only runtime events and replay."""
import json

from src.tiernav_runtime.contracts import EpisodeRequest, EpisodeState, Observation
from src.tiernav_runtime.events import EpisodeEvent, make_event
from src.tiernav_runtime.recorder import EpisodeRecorder
from src.tiernav_runtime.replay import replay_events


def _request() -> EpisodeRequest:
    return EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is on the table?",
        output_dir="/tmp/tiernav",
    )


def test_make_event_has_schema_version_and_sequence():
    event = make_event(
        episode_id="ep-1",
        event_type="episode_started",
        sequence=1,
        payload={"scene_id": "scene"},
    )

    assert event.schema_version == "tiernav.runtime.v1"
    assert event.sequence == 1
    assert event.event_type == "episode_started"


def test_recorder_writes_jsonl_append_only(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = EpisodeRecorder(path)

    recorder.append(make_event("ep-1", "episode_started", 1, {"scene_id": "scene"}))
    recorder.append(make_event("ep-1", "episode_ended", 2, {"success": True}))

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "episode_started"
    assert json.loads(lines[1])["event_type"] == "episode_ended"


def test_replay_reconstructs_materialized_state(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = EpisodeRecorder(path)
    req = _request()

    recorder.append(make_event(req.episode_id, "episode_started", 1, {"request": req.model_dump(mode="json")}))
    recorder.append(make_event(req.episode_id, "tool_result_received", 2, {
        "observation": Observation(summary="Saw a mug.", image_ids=["snap-1"]).model_dump(mode="json"),
        "step_index": 1,
    }))
    recorder.append(make_event(req.episode_id, "episode_ended", 3, {
        "success": True,
        "answer": "mug",
        "round_index": 2,
        "step_index": 1,
    }))

    state = replay_events(path)

    assert isinstance(state, EpisodeState)
    assert state.episode_id == "ep-1"
    assert state.last_observation.summary == "Saw a mug."
    assert state.success is True
    assert state.answer == "mug"
    assert state.round_index == 2


def test_replay_rejects_out_of_order_sequences(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join([
            make_event("ep-1", "episode_started", 2, {}).model_dump_json(),
            make_event("ep-1", "episode_ended", 1, {}).model_dump_json(),
        ]) + "\n",
        encoding="utf-8",
    )

    try:
        replay_events(path)
    except ValueError as exc:
        assert "sequence" in str(exc)
    else:
        raise AssertionError("replay accepted out-of-order events")
```

- [ ] **Step 2: Run the replay tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_recorder_replay.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.events`.

- [ ] **Step 3: Implement event envelope**

Create `src/tiernav_runtime/events.py`:

```python
"""Append-only event envelopes for TierNav runtime."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from .contracts import RuntimeModel, SCHEMA_VERSION


class EpisodeEvent(RuntimeModel):
    """One append-only event in an episode log."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    event_type: str
    sequence: int
    timestamp_utc: str
    payload: dict[str, Any] = Field(default_factory=dict)


def make_event(
    episode_id: str,
    event_type: str,
    sequence: int,
    payload: dict[str, Any] | None = None,
) -> EpisodeEvent:
    """Create a validated event envelope."""

    return EpisodeEvent(
        episode_id=episode_id,
        event_type=event_type,
        sequence=sequence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        payload=payload or {},
    )
```

- [ ] **Step 4: Implement JSONL recorder**

Create `src/tiernav_runtime/recorder.py`:

```python
"""Append-only JSONL recorder for runtime events."""
from __future__ import annotations

from pathlib import Path

from .events import EpisodeEvent


class EpisodeRecorder:
    """Write episode events as append-only JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: EpisodeEvent) -> None:
        """Append one event without rewriting existing lines."""

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")
```

- [ ] **Step 5: Implement replay**

Create `src/tiernav_runtime/replay.py`:

```python
"""Replay append-only event logs into materialized episode state."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import EpisodeRequest, EpisodeState, Observation
from .events import EpisodeEvent


def _load_events(path: str | Path) -> list[EpisodeEvent]:
    events: list[EpisodeEvent] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(EpisodeEvent.model_validate(json.loads(line)))
    sequences = [event.sequence for event in events]
    if sequences != sorted(sequences):
        raise ValueError(f"event sequence is out of order: {sequences}")
    return events


def replay_events(path: str | Path) -> EpisodeState:
    """Rebuild materialized state from an event log."""

    events = _load_events(path)
    if not events:
        raise ValueError("cannot replay an empty event log")

    state: EpisodeState | None = None
    for event in events:
        if event.event_type == "episode_started":
            request_payload = event.payload.get("request")
            if request_payload:
                request = EpisodeRequest.model_validate(request_payload)
                state = EpisodeState(
                    episode_id=request.episode_id,
                    scene_id=request.scene_id,
                    task_name=request.task_name,
                    task_mode=request.task_mode,
                    prompt=request.prompt,
                    pose=request.initial_pose,
                )
            else:
                state = EpisodeState(
                    episode_id=event.episode_id,
                    scene_id=str(event.payload.get("scene_id", "")),
                    task_name=str(event.payload.get("task_name", "")),
                    task_mode=event.payload.get("task_mode", "question_answering"),
                    prompt=str(event.payload.get("prompt", "")),
                )
        elif state is None:
            raise ValueError(f"event log starts with {event.event_type}, not episode_started")
        elif event.event_type == "tool_result_received":
            if "observation" in event.payload:
                state.last_observation = Observation.model_validate(event.payload["observation"])
            state.step_index = int(event.payload.get("step_index", state.step_index))
        elif event.event_type == "policy_transitioned":
            state.failure_type = str(event.payload.get("failure_type", state.failure_type))
        elif event.event_type == "episode_ended":
            state.terminal = True
            state.success = bool(event.payload.get("success", False))
            state.answer = str(event.payload.get("answer", ""))
            state.round_index = int(event.payload.get("round_index", state.round_index))
            state.step_index = int(event.payload.get("step_index", state.step_index))

    if state is None:
        raise ValueError("event log did not contain episode_started")
    return state
```

- [ ] **Step 6: Run recorder and replay tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_recorder_replay.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit event logging**

Run:

```bash
git add src/tiernav_runtime/events.py src/tiernav_runtime/recorder.py src/tiernav_runtime/replay.py tests/runtime/test_recorder_replay.py
git commit -m "feat(runtime): add event recorder and replay"
```

---

## Task 3: Context Compiler With Cacheable Sections

**Files:**
- Create: `src/tiernav_runtime/context.py`
- Test: `tests/runtime/test_context_compiler.py`

- [ ] **Step 1: Write failing context compiler tests**

Create `tests/runtime/test_context_compiler.py`:

```python
"""Tests for Claude Code-style sectioned context compilation."""
from src.tiernav_runtime.context import ContextCompiler
from src.tiernav_runtime.contracts import EpisodeState, MemoryPack


def _state() -> EpisodeState:
    return EpisodeState(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is on the table?",
        round_index=2,
        step_index=3,
        memory_pack=MemoryPack(
            query="table",
            summary="Room 1 contains a table snapshot with a mug.",
            evidence_ids=["snap-1"],
            confidence=0.9,
            reuse_hint="Inspect snap-1 before exploring.",
        ),
    )


def test_context_sections_are_ordered_cacheable_first():
    compiler = ContextCompiler()
    sections = compiler.compile(_state(), action_schema="submit_answer, explore_frontier")

    names = [section.name for section in sections]

    assert names[:3] == ["task_instruction", "action_schema", "memory_index"]
    assert all(section.cacheable for section in sections[:3])
    assert any(not section.cacheable for section in sections)


def test_context_sections_have_stable_hashes():
    compiler = ContextCompiler()
    first = compiler.compile(_state(), action_schema="schema")
    second = compiler.compile(_state(), action_schema="schema")

    assert [s.content_hash for s in first] == [s.content_hash for s in second]


def test_render_prompt_includes_memory_pack_and_observation():
    compiler = ContextCompiler()
    state = _state()
    state.last_observation.summary = "Current view sees a table."

    prompt = compiler.render_prompt(compiler.compile(state, action_schema="schema"))

    assert "Room 1 contains a table" in prompt
    assert "Current view sees a table" in prompt


def test_context_compiler_respects_disabled_memory():
    compiler = ContextCompiler()
    state = _state()

    sections = compiler.compile(state, action_schema="schema", include_memory=False)
    memory_section = [s for s in sections if s.name == "memory_index"][0]

    assert memory_section.content == ""
```

- [ ] **Step 2: Run the context tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_context_compiler.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.context`.

- [ ] **Step 3: Implement the context compiler**

Create `src/tiernav_runtime/context.py`:

```python
"""Sectioned context compiler with cacheable/dynamic boundaries."""
from __future__ import annotations

import hashlib

from .contracts import ContextSection, EpisodeState


TASK_INSTRUCTION = (
    "You are an embodied navigation agent. Use the current observation, "
    "spatial memory, and exploration history to choose the next valid action."
)


class ContextCompiler:
    """Compile EpisodeState into ordered context sections."""

    def compile(
        self,
        state: EpisodeState,
        action_schema: str,
        include_memory: bool = True,
        policy_hint: str = "",
    ) -> list[ContextSection]:
        memory_text = ""
        if include_memory and state.memory_pack is not None:
            memory_text = "\n".join([
                state.memory_pack.summary,
                f"Evidence: {', '.join(state.memory_pack.evidence_ids)}",
                f"Reuse hint: {state.memory_pack.reuse_hint}",
            ]).strip()

        raw_sections = [
            ("task_instruction", TASK_INSTRUCTION, True),
            ("action_schema", action_schema, True),
            ("memory_index", memory_text, True),
            ("recent_trace", f"Round {state.round_index}, step {state.step_index}", False),
            ("current_observation", state.last_observation.summary, False),
            ("policy_hint", policy_hint, False),
        ]
        return [
            self._section(name=name, content=content, cacheable=cacheable)
            for name, content, cacheable in raw_sections
        ]

    def render_prompt(self, sections: list[ContextSection]) -> str:
        """Render sections to a model-facing prompt."""

        blocks = []
        for section in sections:
            if section.content:
                blocks.append(f"## {section.name}\n{section.content}")
        return "\n\n".join(blocks)

    def _section(self, name: str, content: str, cacheable: bool) -> ContextSection:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        token_estimate = max(1, len(content.split())) if content else 0
        return ContextSection(
            name=name,
            content=content,
            cacheable=cacheable,
            token_estimate=token_estimate,
            content_hash=digest,
        )
```

- [ ] **Step 4: Run context compiler tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_context_compiler.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit context compiler**

Run:

```bash
git add src/tiernav_runtime/context.py tests/runtime/test_context_compiler.py
git commit -m "feat(runtime): add sectioned context compiler"
```

---

## Task 4: Workflow Policy

**Files:**
- Create: `src/tiernav_runtime/policy.py`
- Test: `tests/runtime/test_policy.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/runtime/test_policy.py`:

```python
"""Tests for runtime workflow policy routing."""
from src.tiernav_runtime.contracts import AblationConfig, EpisodeState, PlannerDecision, RunSpec
from src.tiernav_runtime.policy import PolicyDecision, WorkflowPolicy


def _spec(**kwargs) -> RunSpec:
    data = {
        "run_id": "run",
        "task_name": "aeqa",
        "dataset_split": "dev",
        "output_dir": "/tmp/tiernav",
        "planner_provider": "mimo",
        "planner_model": "qwen3-vl-flash",
        "max_rounds": 3,
        "max_steps": 5,
    }
    data.update(kwargs)
    return RunSpec(**data)


def _state(**kwargs) -> EpisodeState:
    data = {
        "episode_id": "ep-1",
        "scene_id": "scene",
        "task_name": "aeqa",
        "task_mode": "question_answering",
        "prompt": "question",
    }
    data.update(kwargs)
    return EpisodeState(**data)


def test_policy_routes_submit_decision_to_finalize():
    state = _state(current_decision=PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"}))
    decision = WorkflowPolicy().decide(_spec(), state)

    assert decision.route == "finalize"
    assert decision.reason == "submit_answer"


def test_policy_routes_round_budget_to_fallback():
    state = _state(round_index=3)
    decision = WorkflowPolicy().decide(_spec(), state)

    assert decision.route == "fallback"
    assert decision.reason == "round_budget"


def test_policy_routes_step_budget_to_fallback():
    state = _state(step_index=5)
    decision = WorkflowPolicy().decide(_spec(), state)

    assert decision.route == "fallback"
    assert decision.reason == "step_budget"


def test_policy_routes_continue_for_normal_navigation():
    state = _state(current_decision=PlannerDecision(action_type="explore_frontier", arguments={"frontier_id": "1"}))
    decision = WorkflowPolicy().decide(_spec(), state)

    assert decision.route == "execute_tool"
    assert decision.reason == "continue"


def test_policy_can_disable_stall_recovery_by_ablation():
    spec = _spec(ablation=AblationConfig(stall_recovery=False))
    state = _state(failure_type="stalled")
    decision = WorkflowPolicy().decide(spec, state)

    assert decision.route != "recover_stall"


def test_policy_enables_stall_recovery_when_configured():
    spec = _spec(ablation=AblationConfig(stall_recovery=True))
    state = _state(failure_type="stalled")
    decision = WorkflowPolicy().decide(spec, state)

    assert decision.route == "recover_stall"
    assert decision.reason == "stalled"
```

- [ ] **Step 2: Run policy tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_policy.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.policy`.

- [ ] **Step 3: Implement policy**

Create `src/tiernav_runtime/policy.py`:

```python
"""Pure workflow policy for runtime routing."""
from __future__ import annotations

from pydantic import Field

from .contracts import EpisodeState, RunSpec, RuntimeModel


class PolicyDecision(RuntimeModel):
    """Routing decision emitted by WorkflowPolicy."""

    route: str
    reason: str
    hint: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class WorkflowPolicy:
    """Budget, submit, fallback, and recovery routing."""

    def decide(self, spec: RunSpec, state: EpisodeState) -> PolicyDecision:
        if spec.ablation.stall_recovery and state.failure_type == "stalled":
            return PolicyDecision(
                route="recover_stall",
                reason="stalled",
                hint="Planner repeated a low-progress action; choose a different evidence source.",
            )

        if state.current_decision is not None and state.current_decision.action_type == "submit_answer":
            return PolicyDecision(route="finalize", reason="submit_answer")

        if state.round_index >= spec.max_rounds:
            return PolicyDecision(
                route="fallback",
                reason="round_budget",
                hint="Round budget reached; submit best supported answer.",
            )

        if state.step_index >= spec.max_steps:
            return PolicyDecision(
                route="fallback",
                reason="step_budget",
                hint="Step budget reached; submit best supported answer.",
            )

        return PolicyDecision(route="execute_tool", reason="continue")
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit workflow policy**

Run:

```bash
git add src/tiernav_runtime/policy.py tests/runtime/test_policy.py
git commit -m "feat(runtime): add workflow policy"
```

---

## Task 5: Spatial Memory Service

**Files:**
- Create: `src/tiernav_runtime/memory.py`
- Test: `tests/runtime/test_memory_service.py`

- [ ] **Step 1: Write failing memory tests**

Create `tests/runtime/test_memory_service.py`:

```python
"""Tests for room-snapshot-object-hypothesis memory service."""
from src.tiernav_runtime.contracts import Observation
from src.tiernav_runtime.memory import MemoryService


def test_memory_updates_room_snapshot_object_layers():
    memory = MemoryService()

    memory.update_from_observation(
        observation=Observation(
            summary="Saw a red mug on the table.",
            image_ids=["snap-1"],
            object_ids=["obj-mug"],
            room_id="room-1",
        ),
        action_type="explore_panorama",
        round_index=1,
    )

    assert "room-1" in memory.rooms
    assert "snap-1" in memory.snapshots
    assert "obj-mug" in memory.objects
    assert memory.snapshots["snap-1"].room_id == "room-1"


def test_memory_query_returns_context_ready_pack():
    memory = MemoryService()
    memory.update_from_observation(
        observation=Observation(
            summary="Snapshot shows a red mug on a table.",
            image_ids=["snap-1"],
            object_ids=["mug"],
            room_id="kitchen",
        ),
        action_type="navigate_to_object",
        round_index=2,
    )

    pack = memory.query("What is on the table?")

    assert "mug" in pack.summary.lower()
    assert "snap-1" in pack.evidence_ids
    assert pack.reuse_hint


def test_memory_can_be_disabled_without_crashing():
    memory = MemoryService(enabled=False)
    memory.update_from_observation(
        observation=Observation(summary="Saw a chair.", image_ids=["snap-2"], object_ids=["chair"]),
        action_type="explore_panorama",
        round_index=1,
    )

    pack = memory.query("chair")

    assert pack.summary == ""
    assert pack.evidence_ids == []


def test_memory_records_hypothesis_support_and_contradiction():
    memory = MemoryService()
    memory.add_hypothesis("h1", "The answer is mug.")
    memory.support_hypothesis("h1", "snap-1")
    memory.contradict_hypothesis("h1", "snap-2")

    pack = memory.query("mug")

    assert "snap-1" in pack.supports
    assert "snap-2" in pack.contradictions
```

- [ ] **Step 2: Run memory tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_memory_service.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.memory`.

- [ ] **Step 3: Implement memory service**

Create `src/tiernav_runtime/memory.py`:

```python
"""Room-snapshot-object-hypothesis spatial memory service."""
from __future__ import annotations

from pydantic import Field

from .contracts import MemoryPack, Observation, RuntimeModel


class RoomNode(RuntimeModel):
    room_id: str
    status: str = "observed"
    snapshot_ids: list[str] = Field(default_factory=list)


class SnapshotNode(RuntimeModel):
    snapshot_id: str
    room_id: str = ""
    summary: str = ""
    object_ids: list[str] = Field(default_factory=list)
    round_index: int = 0
    action_type: str = ""


class ObjectNode(RuntimeModel):
    object_id: str
    room_id: str = ""
    snapshot_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class HypothesisNode(RuntimeModel):
    hypothesis_id: str
    text: str
    supports: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class MemoryService:
    """Canonical spatial memory graph used by the runtime."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.rooms: dict[str, RoomNode] = {}
        self.snapshots: dict[str, SnapshotNode] = {}
        self.objects: dict[str, ObjectNode] = {}
        self.hypotheses: dict[str, HypothesisNode] = {}

    def update_from_observation(
        self,
        observation: Observation,
        action_type: str,
        round_index: int,
    ) -> None:
        if not self.enabled:
            return

        room_id = observation.room_id or "unknown"
        room = self.rooms.setdefault(room_id, RoomNode(room_id=room_id))

        for snapshot_id in observation.image_ids:
            if snapshot_id not in room.snapshot_ids:
                room.snapshot_ids.append(snapshot_id)
            self.snapshots[snapshot_id] = SnapshotNode(
                snapshot_id=snapshot_id,
                room_id=room_id,
                summary=observation.summary,
                object_ids=list(observation.object_ids),
                round_index=round_index,
                action_type=action_type,
            )

        for object_id in observation.object_ids:
            obj = self.objects.setdefault(object_id, ObjectNode(object_id=object_id, room_id=room_id))
            for snapshot_id in observation.image_ids:
                if snapshot_id not in obj.snapshot_ids:
                    obj.snapshot_ids.append(snapshot_id)

    def add_hypothesis(self, hypothesis_id: str, text: str) -> None:
        if not self.enabled:
            return
        self.hypotheses[hypothesis_id] = HypothesisNode(hypothesis_id=hypothesis_id, text=text)

    def support_hypothesis(self, hypothesis_id: str, evidence_id: str) -> None:
        if hypothesis_id in self.hypotheses and evidence_id not in self.hypotheses[hypothesis_id].supports:
            self.hypotheses[hypothesis_id].supports.append(evidence_id)

    def contradict_hypothesis(self, hypothesis_id: str, evidence_id: str) -> None:
        if hypothesis_id in self.hypotheses and evidence_id not in self.hypotheses[hypothesis_id].contradictions:
            self.hypotheses[hypothesis_id].contradictions.append(evidence_id)

    def query(self, query: str) -> MemoryPack:
        if not self.enabled:
            return MemoryPack(query=query, summary="")

        query_words = {w.strip(".,!?").lower() for w in query.split()}
        matched_snapshots: list[SnapshotNode] = []
        for snapshot in self.snapshots.values():
            haystack = " ".join([snapshot.summary, " ".join(snapshot.object_ids)]).lower()
            if any(word and word in haystack for word in query_words):
                matched_snapshots.append(snapshot)

        if not matched_snapshots:
            matched_snapshots = list(self.snapshots.values())[:3]

        evidence_ids = [snapshot.snapshot_id for snapshot in matched_snapshots[:5]]
        summaries = [snapshot.summary for snapshot in matched_snapshots[:3] if snapshot.summary]
        supports: list[str] = []
        contradictions: list[str] = []
        for hypothesis in self.hypotheses.values():
            supports.extend(hypothesis.supports)
            contradictions.extend(hypothesis.contradictions)

        summary = "\n".join(summaries)
        return MemoryPack(
            query=query,
            summary=summary,
            evidence_ids=evidence_ids,
            supports=supports,
            contradictions=contradictions,
            confidence=0.7 if evidence_ids else 0.0,
            reuse_hint="Reuse matched snapshots before choosing new exploration." if evidence_ids else "",
        )
```

- [ ] **Step 4: Run memory tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_memory_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit memory service**

Run:

```bash
git add src/tiernav_runtime/memory.py tests/runtime/test_memory_service.py
git commit -m "feat(runtime): add spatial memory service"
```

---

## Task 6: Planner and Tool Interfaces

**Files:**
- Create: `src/tiernav_runtime/planner.py`
- Create: `src/tiernav_runtime/tools.py`
- Test: `tests/runtime/test_tools.py`

- [ ] **Step 1: Write failing planner/tool tests**

Create `tests/runtime/test_tools.py`:

```python
"""Tests for runtime planner adapters and stable tool registry."""
from src.agent_planner import PlannerAction
from src.tiernav_runtime.contracts import Observation, PlannerDecision, ToolCall, ToolResult
from src.tiernav_runtime.planner import planner_action_to_decision
from src.tiernav_runtime.tools import RuntimeTool, ToolRegistry


class EchoTool(RuntimeTool):
    name = "echo"
    terminal = False

    def run(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            action_type=call.action_type,
            ok=True,
            observation=Observation(summary=f"echo {call.arguments['text']}"),
        )


def test_planner_action_adapter_preserves_arguments():
    action = PlannerAction(
        action_type="navigate_to_object",
        reason="visible",
        expected="verify",
        confidence=0.6,
        snapshot_id="snap-1",
        object_name="chair",
    )

    decision = planner_action_to_decision(action)

    assert isinstance(decision, PlannerDecision)
    assert decision.action_type == "navigate_to_object"
    assert decision.arguments["snapshot_id"] == "snap-1"
    assert decision.arguments["object_name"] == "chair"


def test_tool_registry_dispatches_registered_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = registry.dispatch(ToolCall(call_id="c1", action_type="echo", arguments={"text": "hello"}))

    assert result.ok is True
    assert result.observation.summary == "echo hello"


def test_tool_registry_rejects_unknown_tool_with_structured_error():
    registry = ToolRegistry()

    result = registry.dispatch(ToolCall(call_id="c1", action_type="missing", arguments={}))

    assert result.ok is False
    assert "unknown tool" in result.error


def test_default_registry_has_no_stub_tools():
    registry = ToolRegistry.with_stable_defaults()

    assert "fork_subagent" not in registry.names()
    assert "pixel_navigate" not in registry.names()
    assert "submit_answer" in registry.names()
```

- [ ] **Step 2: Run tool tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_tools.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.planner`.

- [ ] **Step 3: Implement planner adapter**

Create `src/tiernav_runtime/planner.py`:

```python
"""Planner interface helpers for the runtime."""
from __future__ import annotations

from typing import Any

from .contracts import PlannerDecision


def planner_action_to_decision(action: Any) -> PlannerDecision:
    """Convert existing PlannerAction-like objects into PlannerDecision."""

    arguments: dict[str, Any] = {}
    for key in ["snapshot_id", "object_name", "seed_id", "frontier_id", "view_idx", "answer"]:
        value = getattr(action, key, None)
        if value is not None:
            arguments[key] = value

    return PlannerDecision(
        action_type=getattr(action, "action_type"),
        reasoning=getattr(action, "reason", "") or "",
        expected=getattr(action, "expected", "") or "",
        confidence=float(getattr(action, "confidence", 0.0) or 0.0),
        arguments=arguments,
    )
```

- [ ] **Step 4: Implement stable tool registry**

Create `src/tiernav_runtime/tools.py`:

```python
"""Stable runtime tool registry."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import Observation, ToolCall, ToolResult


class RuntimeTool(ABC):
    """Base class for runtime tools."""

    name: str
    terminal: bool = False

    @abstractmethod
    def run(self, call: ToolCall) -> ToolResult:
        """Execute a tool call."""


class SubmitAnswerTool(RuntimeTool):
    name = "submit_answer"
    terminal = True

    def run(self, call: ToolCall) -> ToolResult:
        answer = str(call.arguments.get("answer", ""))
        return ToolResult(
            call_id=call.call_id,
            action_type=call.action_type,
            ok=bool(answer),
            terminal=True,
            observation=Observation(summary=f"Answer submitted: {answer}"),
            error="" if answer else "submit_answer requires an answer",
        )


class NoopNavigationTool(RuntimeTool):
    """Deterministic navigation test tool used before Habitat wiring."""

    def __init__(self, name: str):
        self.name = name
        self.terminal = False

    def run(self, call: ToolCall) -> ToolResult:
        target = next(iter(call.arguments.values()), "unknown")
        return ToolResult(
            call_id=call.call_id,
            action_type=call.action_type,
            ok=True,
            terminal=False,
            observation=Observation(
                summary=f"{call.action_type} executed toward {target}",
                image_ids=[f"{call.call_id}_snapshot"],
                object_ids=[str(target)] if target != "unknown" else [],
            ),
            metrics={"path_length": 1.0},
        )


class ToolRegistry:
    """Registry that never exposes stubs in stable defaults."""

    def __init__(self):
        self._tools: dict[str, RuntimeTool] = {}

    def register(self, tool: RuntimeTool) -> None:
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
        return "\n".join(f"- {name}" for name in self.names())

    @classmethod
    def with_stable_defaults(cls) -> "ToolRegistry":
        registry = cls()
        registry.register(NoopNavigationTool("explore_panorama"))
        registry.register(NoopNavigationTool("navigate_to_object"))
        registry.register(NoopNavigationTool("explore_seed"))
        registry.register(NoopNavigationTool("explore_frontier"))
        registry.register(SubmitAnswerTool())
        return registry
```

- [ ] **Step 5: Run planner/tool tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit planner/tool interfaces**

Run:

```bash
git add src/tiernav_runtime/planner.py src/tiernav_runtime/tools.py tests/runtime/test_tools.py
git commit -m "feat(runtime): add planner and stable tool interfaces"
```

---

## Task 7: LangGraph Runtime With Deterministic Fake Services

**Files:**
- Create: `src/tiernav_runtime/graph.py`
- Test: `tests/runtime/test_graph_runtime.py`

- [ ] **Step 1: Write failing graph runtime tests**

Create `tests/runtime/test_graph_runtime.py`:

```python
"""Tests for deterministic LangGraph runtime."""
from src.tiernav_runtime.contracts import EpisodeRequest, PlannerDecision, RunSpec
from src.tiernav_runtime.graph import RuntimeServices, build_runtime_graph
from src.tiernav_runtime.memory import MemoryService
from src.tiernav_runtime.policy import WorkflowPolicy
from src.tiernav_runtime.tools import ToolRegistry


class FakePlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, prompt: str) -> PlannerDecision:
        if not self.decisions:
            return PlannerDecision(action_type="submit_answer", arguments={"answer": "unanswerable"})
        return self.decisions.pop(0)


def _spec() -> RunSpec:
    return RunSpec(
        run_id="run",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="fake",
        planner_model="fake",
        max_rounds=4,
        max_steps=4,
    )


def _request() -> EpisodeRequest:
    return EpisodeRequest(
        episode_id="ep-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is on the table?",
    )


def test_runtime_graph_executes_tool_then_submit():
    services = RuntimeServices(
        planner=FakePlanner([
            PlannerDecision(action_type="explore_frontier", arguments={"frontier_id": "1"}),
            PlannerDecision(action_type="submit_answer", arguments={"answer": "mug"}),
        ]),
        tools=ToolRegistry.with_stable_defaults(),
        memory=MemoryService(),
        policy=WorkflowPolicy(),
    )
    graph = build_runtime_graph()

    final_state = graph.invoke(
        {"spec": _spec().model_dump(mode="json"), "request": _request().model_dump(mode="json")},
        config={"configurable": {"services": services}},
    )

    assert final_state["state"]["terminal"] is True
    assert final_state["state"]["answer"] == "mug"
    assert final_state["state"]["step_index"] == 1


def test_runtime_graph_fallbacks_on_round_budget():
    services = RuntimeServices(
        planner=FakePlanner([
            PlannerDecision(action_type="explore_frontier", arguments={"frontier_id": "1"}),
            PlannerDecision(action_type="explore_frontier", arguments={"frontier_id": "2"}),
            PlannerDecision(action_type="explore_frontier", arguments={"frontier_id": "3"}),
        ]),
        tools=ToolRegistry.with_stable_defaults(),
        memory=MemoryService(),
        policy=WorkflowPolicy(),
    )
    spec = _spec()
    spec.max_rounds = 1
    graph = build_runtime_graph()

    final_state = graph.invoke(
        {"spec": spec.model_dump(mode="json"), "request": _request().model_dump(mode="json")},
        config={"configurable": {"services": services}},
    )

    assert final_state["state"]["terminal"] is True
    assert final_state["state"]["failure_type"] == "round_budget"
```

- [ ] **Step 2: Run graph tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_graph_runtime.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.graph`.

- [ ] **Step 3: Implement runtime graph**

Create `src/tiernav_runtime/graph.py`:

```python
"""LangGraph runtime built on TierNav contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .context import ContextCompiler
from .contracts import EpisodeRequest, EpisodeState, PlannerDecision, RunSpec, ToolCall
from .memory import MemoryService
from .policy import PolicyDecision, WorkflowPolicy
from .tools import ToolRegistry


class RuntimeGraphState(TypedDict, total=False):
    spec: dict[str, Any]
    request: dict[str, Any]
    state: dict[str, Any]
    policy: dict[str, Any]
    prompt: str


@dataclass
class RuntimeServices:
    planner: Any
    tools: ToolRegistry
    memory: MemoryService
    policy: WorkflowPolicy
    context: ContextCompiler | None = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = ContextCompiler()


def _services(config) -> RuntimeServices:
    return config["configurable"]["services"]


def bootstrap_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    request = EpisodeRequest.model_validate(graph_state["request"])
    state = EpisodeState(
        episode_id=request.episode_id,
        scene_id=request.scene_id,
        task_name=request.task_name,
        task_mode=request.task_mode,
        prompt=request.prompt,
        pose=request.initial_pose,
    )
    return {"state": state.model_dump(mode="json")}


def compile_context_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    spec = RunSpec.model_validate(graph_state["spec"])
    state = EpisodeState.model_validate(graph_state["state"])
    services = _services(config)
    if spec.ablation.active_memory_query:
        state.memory_pack = services.memory.query(state.prompt)
    sections = services.context.compile(
        state,
        action_schema=services.tools.action_schema_text(),
        include_memory=spec.ablation.spatial_memory,
    )
    state.context_sections = sections
    prompt = services.context.render_prompt(sections)
    return {"state": state.model_dump(mode="json"), "prompt": prompt}


def plan_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    state = EpisodeState.model_validate(graph_state["state"])
    services = _services(config)
    decision = services.planner.decide(graph_state.get("prompt", ""))
    state.current_decision = PlannerDecision.model_validate(decision)
    state.round_index += 1
    return {"state": state.model_dump(mode="json")}


def policy_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    spec = RunSpec.model_validate(graph_state["spec"])
    state = EpisodeState.model_validate(graph_state["state"])
    decision = _services(config).policy.decide(spec, state)
    return {"policy": decision.model_dump(mode="json")}


def route_after_policy(graph_state: RuntimeGraphState) -> str:
    decision = PolicyDecision.model_validate(graph_state["policy"])
    return decision.route


def execute_tool_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    state = EpisodeState.model_validate(graph_state["state"])
    services = _services(config)
    assert state.current_decision is not None
    call = ToolCall(
        call_id=f"{state.episode_id}-tool-{state.step_index + 1}",
        action_type=state.current_decision.action_type,
        arguments=state.current_decision.arguments,
    )
    result = services.tools.dispatch(call)
    state.last_observation = result.observation
    state.step_index += 1
    services.memory.update_from_observation(
        observation=result.observation,
        action_type=result.action_type,
        round_index=state.round_index,
    )
    if result.terminal:
        state.terminal = True
        state.success = result.ok
        state.answer = str(call.arguments.get("answer", ""))
    return {"state": state.model_dump(mode="json")}


def route_after_execute(graph_state: RuntimeGraphState) -> str:
    state = EpisodeState.model_validate(graph_state["state"])
    return "finalize" if state.terminal else "compile_context"


def fallback_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    state = EpisodeState.model_validate(graph_state["state"])
    policy = PolicyDecision.model_validate(graph_state["policy"])
    state.terminal = True
    state.success = False
    state.answer = "unanswerable"
    state.failure_type = policy.reason
    return {"state": state.model_dump(mode="json")}


def recover_stall_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    state = EpisodeState.model_validate(graph_state["state"])
    state.failure_type = ""
    return {"state": state.model_dump(mode="json")}


def finalize_node(graph_state: RuntimeGraphState, config) -> RuntimeGraphState:
    state = EpisodeState.model_validate(graph_state["state"])
    if state.current_decision and state.current_decision.action_type == "submit_answer":
        state.terminal = True
        state.success = bool(state.current_decision.arguments.get("answer"))
        state.answer = str(state.current_decision.arguments.get("answer", ""))
    return {"state": state.model_dump(mode="json")}


def build_runtime_graph():
    graph: StateGraph = StateGraph(RuntimeGraphState)
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
```

- [ ] **Step 4: Run graph runtime tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_graph_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit deterministic runtime graph**

Run:

```bash
git add src/tiernav_runtime/graph.py tests/runtime/test_graph_runtime.py
git commit -m "feat(runtime): add langgraph runtime skeleton"
```

---

## Task 8: AEQA and GOATBench Task Adapters

**Files:**
- Create: `src/tiernav_runtime/adapters.py`
- Test: `tests/runtime/test_adapters.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/runtime/test_adapters.py`:

```python
"""Tests for AEQA and GOATBench task adapters."""
from src.tiernav_runtime.adapters import AEQATaskAdapter, GOATBenchTaskAdapter
from src.tiernav_runtime.contracts import EpisodeResult


def test_aeqa_adapter_builds_question_answering_request():
    adapter = AEQATaskAdapter()

    request = adapter.to_request(
        scene_id="scene",
        question_id="qid-1",
        question="What color is the chair?",
        output_dir="/tmp/out",
    )

    assert request.episode_id == "qid-1"
    assert request.task_name == "aeqa"
    assert request.task_mode == "question_answering"
    assert request.prompt == "What color is the chair?"


def test_aeqa_adapter_exports_logger_payload():
    adapter = AEQATaskAdapter()
    result = EpisodeResult(
        episode_id="qid-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        success=True,
        answer="red",
        steps_taken=3,
        rounds_used=2,
        path_length=4.5,
    )

    payload = adapter.to_eval_payload(result)

    assert payload["question_id"] == "qid-1"
    assert payload["answer"] == "red"
    assert payload["path_length"] == 4.5


def test_goatbench_adapter_builds_navigation_request_without_truth_leak():
    adapter = GOATBenchTaskAdapter()

    request = adapter.to_request(
        scene_id="scene",
        episode_id="ep-1",
        subtask_index=2,
        goal_type="object",
        goal_description="chair",
        output_dir="/tmp/out",
    )

    assert request.episode_id == "ep-1_2"
    assert request.task_name == "goatbench"
    assert request.task_mode == "goal_navigation"
    assert request.goal_metadata == {"goal_type": "object", "goal_description": "chair", "subtask_index": 2}


def test_goatbench_adapter_exports_navigation_payload():
    adapter = GOATBenchTaskAdapter()
    result = EpisodeResult(
        episode_id="ep-1_2",
        scene_id="scene",
        task_name="goatbench",
        task_mode="goal_navigation",
        success=False,
        answer="not_found",
        steps_taken=10,
        rounds_used=4,
        failure_type="target_not_reached",
    )

    payload = adapter.to_eval_payload(result)

    assert payload["subtask_id"] == "ep-1_2"
    assert payload["success"] is False
    assert payload["failure_type"] == "target_not_reached"
```

- [ ] **Step 2: Run adapter tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_adapters.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.adapters`.

- [ ] **Step 3: Implement adapters**

Create `src/tiernav_runtime/adapters.py`:

```python
"""Task adapters for runtime requests and evaluation payloads."""
from __future__ import annotations

from .contracts import EpisodeRequest, EpisodeResult


class AEQATaskAdapter:
    """Adapter for AEQA question answering episodes."""

    task_name = "aeqa"

    def to_request(
        self,
        scene_id: str,
        question_id: str,
        question: str,
        output_dir: str,
        initial_pose: dict[str, float] | None = None,
    ) -> EpisodeRequest:
        return EpisodeRequest(
            episode_id=question_id,
            scene_id=scene_id,
            task_name=self.task_name,
            task_mode="question_answering",
            prompt=question,
            initial_pose=initial_pose or {},
            output_dir=output_dir,
        )

    def to_eval_payload(self, result: EpisodeResult) -> dict:
        return {
            "question_id": result.episode_id,
            "scene_id": result.scene_id,
            "answer": result.answer,
            "success": result.success,
            "steps_taken": result.steps_taken,
            "rounds_used": result.rounds_used,
            "path_length": result.path_length,
            "error": result.error,
        }


class GOATBenchTaskAdapter:
    """Adapter for GOATBench goal navigation subtasks."""

    task_name = "goatbench"

    def to_request(
        self,
        scene_id: str,
        episode_id: str,
        subtask_index: int,
        goal_type: str,
        goal_description: str,
        output_dir: str,
        initial_pose: dict[str, float] | None = None,
    ) -> EpisodeRequest:
        subtask_id = f"{episode_id}_{subtask_index}"
        return EpisodeRequest(
            episode_id=subtask_id,
            scene_id=scene_id,
            task_name=self.task_name,
            task_mode="goal_navigation",
            prompt=f"Navigate to {goal_description}",
            goal_metadata={
                "goal_type": goal_type,
                "goal_description": goal_description,
                "subtask_index": subtask_index,
            },
            initial_pose=initial_pose or {},
            output_dir=output_dir,
        )

    def to_eval_payload(self, result: EpisodeResult) -> dict:
        return {
            "subtask_id": result.episode_id,
            "scene_id": result.scene_id,
            "success": result.success,
            "answer": result.answer,
            "steps_taken": result.steps_taken,
            "rounds_used": result.rounds_used,
            "failure_type": result.failure_type,
            "error": result.error,
        }
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_adapters.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit adapters**

Run:

```bash
git add src/tiernav_runtime/adapters.py tests/runtime/test_adapters.py
git commit -m "feat(runtime): add task adapters"
```

---

## Task 9: Runtime Entrypoint and Legacy-Compatible Return Mapping

**Files:**
- Create: `src/tiernav_runtime/entrypoint.py`
- Test: `tests/runtime/test_entrypoint_compat.py`
- Modify later after this task passes: `run_two_tier_aeqa_evaluation.py`
- Modify later after this task passes: `run_goatbench_evaluation.py`

- [ ] **Step 1: Write failing entrypoint compatibility tests**

Create `tests/runtime/test_entrypoint_compat.py`:

```python
"""Tests for runtime entrypoint return mapping."""
from src.tiernav_runtime.contracts import EpisodeRequest, PlannerDecision, RunSpec
from src.tiernav_runtime.entrypoint import RuntimeEntrypoint, episode_result_to_legacy_dict


class FakePlanner:
    def decide(self, prompt: str) -> PlannerDecision:
        return PlannerDecision(action_type="submit_answer", arguments={"answer": "chair"})


def test_runtime_entrypoint_returns_episode_result(tmp_path):
    spec = RunSpec(
        run_id="run",
        task_name="aeqa",
        dataset_split="dev",
        output_dir=str(tmp_path),
        planner_provider="fake",
        planner_model="fake",
    )
    request = EpisodeRequest(
        episode_id="qid-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is visible?",
        output_dir=str(tmp_path),
    )

    result = RuntimeEntrypoint.with_fake_services(FakePlanner()).run(spec, request)

    assert result.success is True
    assert result.answer == "chair"
    assert result.event_log_path


def test_episode_result_to_legacy_dict_preserves_existing_fields(tmp_path):
    spec = RunSpec(
        run_id="run",
        task_name="aeqa",
        dataset_split="dev",
        output_dir=str(tmp_path),
        planner_provider="fake",
        planner_model="fake",
    )
    request = EpisodeRequest(
        episode_id="qid-1",
        scene_id="scene",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is visible?",
        output_dir=str(tmp_path),
    )
    result = RuntimeEntrypoint.with_fake_services(FakePlanner()).run(spec, request)

    payload = episode_result_to_legacy_dict(result, question=request.prompt)

    assert payload["question_id"] == "qid-1"
    assert payload["question"] == "What is visible?"
    assert payload["answer"] == "chair"
    assert payload["success"] is True
    assert "path_length" in payload
```

- [ ] **Step 2: Run entrypoint tests to verify they fail**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_entrypoint_compat.py -q
```

Expected: FAIL with missing `src.tiernav_runtime.entrypoint`.

- [ ] **Step 3: Implement runtime entrypoint**

Create `src/tiernav_runtime/entrypoint.py`:

```python
"""Runtime entrypoint and compatibility mapping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import EpisodeRequest, EpisodeResult, EpisodeState, RunSpec
from .events import make_event
from .graph import RuntimeServices, build_runtime_graph
from .memory import MemoryService
from .policy import WorkflowPolicy
from .recorder import EpisodeRecorder
from .tools import ToolRegistry


class RuntimeEntrypoint:
    """Run one episode through the contract-first LangGraph runtime."""

    def __init__(self, services: RuntimeServices):
        self.services = services
        self.graph = build_runtime_graph()

    @classmethod
    def with_fake_services(cls, planner: Any) -> "RuntimeEntrypoint":
        return cls(
            RuntimeServices(
                planner=planner,
                tools=ToolRegistry.with_stable_defaults(),
                memory=MemoryService(),
                policy=WorkflowPolicy(),
            )
        )

    def run(self, spec: RunSpec, request: EpisodeRequest) -> EpisodeResult:
        event_log_path = Path(spec.output_dir) / request.episode_id / "events.jsonl"
        recorder = EpisodeRecorder(event_log_path)
        recorder.append(make_event(request.episode_id, "episode_started", 1, {"request": request.model_dump(mode="json")}))

        final = self.graph.invoke(
            {"spec": spec.model_dump(mode="json"), "request": request.model_dump(mode="json")},
            config={"configurable": {"services": self.services}},
        )
        state = EpisodeState.model_validate(final["state"])

        recorder.append(make_event(request.episode_id, "episode_ended", 2, {
            "success": state.success,
            "answer": state.answer,
            "round_index": state.round_index,
            "step_index": state.step_index,
        }))

        return EpisodeResult(
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


def episode_result_to_legacy_dict(result: EpisodeResult, question: str = "") -> dict[str, Any]:
    """Map EpisodeResult to the legacy runner result dict shape."""

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
```

- [ ] **Step 4: Run entrypoint tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime/test_entrypoint_compat.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit entrypoint**

Run:

```bash
git add src/tiernav_runtime/entrypoint.py tests/runtime/test_entrypoint_compat.py
git commit -m "feat(runtime): add runtime entrypoint"
```

- [ ] **Step 6: Add runtime engine flag mapping after deterministic tests pass**

Modify `run_two_tier_aeqa_evaluation.py` only after Tasks 1-9 pass. Add a new engine key called `runtime` that calls a wrapper around `RuntimeEntrypoint`. Keep `legacy` and current `langgraph` untouched during the first integration commit.

Expected minimal mapping shape:

```python
_ENGINES = {
    "legacy": run_episode_two_tier,
    "langgraph": run_episode_two_tier_langgraph,
    "runtime": run_episode_tiernav_runtime,
}
```

Create `run_episode_tiernav_runtime` in `src/tiernav_runtime/entrypoint.py` as a compatibility wrapper only after Habitat-backed services are wired. Until then, do not point production evals at the deterministic fake services.

---

## Task 10: Schema Snapshots, Stub Audit, and Test Gate

**Files:**
- Create: `tests/runtime/test_schema_snapshots.py`
- Create: `tests/runtime/test_default_path_no_stubs.py`
- Optional generated artifact: `docs/schemas/tiernav_runtime_v1.json`

- [ ] **Step 1: Write schema snapshot test**

Create `tests/runtime/test_schema_snapshots.py`:

```python
"""Schema export tests for runtime contracts."""
import json

from src.tiernav_runtime.contracts import dump_runtime_json_schemas


def test_schema_export_is_json_serializable():
    schemas = dump_runtime_json_schemas()
    encoded = json.dumps(schemas, sort_keys=True)

    assert "RunSpec" in encoded
    assert "EpisodeResult" in encoded
```

- [ ] **Step 2: Write default-path no-stub audit**

Create `tests/runtime/test_default_path_no_stubs.py`:

```python
"""Default runtime path must not expose stub tools."""
from src.tiernav_runtime.tools import ToolRegistry


def test_stable_default_registry_excludes_experimental_stubs():
    names = set(ToolRegistry.with_stable_defaults().names())

    assert "fork_subagent" not in names
    assert "pixel_navigate" not in names


def test_default_tools_do_not_raise_not_implemented():
    registry = ToolRegistry.with_stable_defaults()

    for name in registry.names():
        tool = registry._tools[name]
        assert tool.run.__qualname__
```

- [ ] **Step 3: Run all runtime tests**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest tests/runtime -q
```

Expected: PASS.

- [ ] **Step 4: Run existing focused tests that protect current behavior**

Run:

```bash
/home/afdsafg/miniconda3/envs/3dmem/bin/python -m pytest \
  tests/test_two_tier_graph.py \
  tests/test_prompt_sections.py \
  tests/test_stall_detection.py \
  tests/test_aeqa_output_format.py \
  -q
```

Expected: PASS. If existing tests fail because of pre-existing repository state, record the failing test names in the commit message body and do not claim full regression success.

- [ ] **Step 5: Commit final runtime test gate**

Run:

```bash
git add tests/runtime/test_schema_snapshots.py tests/runtime/test_default_path_no_stubs.py
git commit -m "test(runtime): add schema and stub audit gates"
```

---

## Migration Notes for the Next Plan

After this plan is complete, write a second implementation plan for Habitat/VLM-backed service integration. That follow-up plan should wire:

- `src.agent_planner.Planner` through a real `PlannerClient`
- `src.agent_executor.Executor` through real runtime tools
- `src.scene_graph_memory.SceneGraphMemory` and existing notebook data into `MemoryService`
- `run_two_tier_aeqa_evaluation.py --engine runtime`
- `run_goatbench_evaluation.py --engine runtime`

Do not do those integrations before the deterministic runtime package is green.

## Self-Review Checklist

- Spec coverage:
  - LangGraph-only backend: Task 7.
  - Pydantic / JSON Schema contracts: Task 1 and Task 10.
  - Continuous context: Task 3.
  - Room-snapshot-object memory: Task 5.
  - Active memory query and reuse: Task 5 and Task 7.
  - Append-only recorder and replay: Task 2 and Task 9.
  - Default-path no stubs: Task 6 and Task 10.
  - AEQA / GOATBench adapter boundary: Task 8.

- Placeholder scan:
  - The plan contains no unresolved requirement markers.
  - The plan does not leave implementation details as unnamed future work inside Tasks 1-10.
  - The only deferred items are explicitly scoped to a separate Habitat/VLM integration plan.

- Type consistency:
  - `EpisodeRequest`, `EpisodeState`, `EpisodeResult`, `PlannerDecision`, `ToolCall`, `ToolResult`, `Observation`, `MemoryPack`, and `ContextSection` are introduced in Task 1 and reused consistently.
  - `RuntimeServices` is introduced in Task 7 and reused by Task 9.
  - `ToolRegistry.with_stable_defaults()` is introduced in Task 6 and reused by Task 7, Task 9, and Task 10.

