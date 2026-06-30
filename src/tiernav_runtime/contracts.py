"""Pydantic contracts for the TierNav runtime."""
from __future__ import annotations

from enum import Enum
from numbers import Real
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator
from typing_extensions import TypeAliasType


SCHEMA_VERSION = "tiernav.runtime.v1"
SchemaVersion = Literal["tiernav.runtime.v1"]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
JsonInt = StrictInt
JsonFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
PoseValues = dict[str, FiniteFloat]
MetricsMap = dict[str, FiniteFloat]
ConfidenceScore = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
JsonScalar = Union[None, bool, str, JsonInt, JsonFloat]
JsonValue = TypeAliasType(
    "JsonValue",
    "Union[None, bool, str, JsonInt, JsonFloat, list[JsonValue], dict[str, JsonValue]]",
)
JsonObject = dict[str, JsonValue]


class RuntimeModel(BaseModel):
    """Base model for runtime contracts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TaskMode(str, Enum):
    QUESTION_ANSWERING = "question_answering"
    GOAL_NAVIGATION = "goal_navigation"


class RuntimeMode(str, Enum):
    """Selects which execution path the runtime entrypoint drives."""

    GRAPH = "graph"
    LEGACY = "legacy"


class AblationConfig(RuntimeModel):
    """Ablation switches for the three main contributions and support levers."""

    continuous_context: bool = True
    spatial_memory: bool = True
    active_memory_query: bool = True
    prompt_cache: bool = True
    stall_recovery: bool = False


class MemoryScope(str, Enum):
    """How far memory reaches across an episode.

    ``PER_QUESTION`` scopes memory to a single question/answer (AEQA: each
    question is scored independently). ``SUBTASK_SEQUENCE`` shares memory
    across the ordered subtasks of one episode (GOATBench: later subtasks may
    reuse observations from earlier ones).
    """

    PER_QUESTION = "per_question"
    SUBTASK_SEQUENCE = "subtask_sequence"


class GoalSpec(RuntimeModel):
    """Goal description for a navigation/scoring episode.

    ``goal_object_ids_for_scoring`` is the authoritative list of target object
    ids used by the scorer, kept separate from ``goal_description`` which is
    what the planner sees in its prompt.
    """

    goal_type: str
    goal_description: str
    goal_object_ids_for_scoring: list[str] = Field(default_factory=list)
    subtask_index: NonNegativeInt = 0
    subtask_total: NonNegativeInt = 0


class BenchmarkRule(RuntimeModel):
    """Per-benchmark success and memory policy."""

    success_distance_m: NonNegativeFloat
    requires_explicit_stop: bool = False
    memory_scope: MemoryScope
    scoring_mode: str
    # Bounded planner retry count on parse/call failures. 0 = no retry
    # (immediate fallback submit). Default 0 preserves current behavior.
    planner_retries: NonNegativeInt = 0


class RunSpec(RuntimeModel):
    """Configuration for a reproducible run or sweep member."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    run_id: str
    task_name: str
    dataset_split: str
    output_dir: str
    runtime_mode: RuntimeMode = RuntimeMode.GRAPH
    planner_provider: str
    planner_model: str
    planner_base_url: str = ""
    planner_api_key_env: str = ""
    seed: NonNegativeInt = 0
    max_rounds: NonNegativeInt = 10
    max_steps: NonNegativeInt = 50
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    metadata: JsonObject = Field(default_factory=dict)


class EpisodeRequest(RuntimeModel):
    """Task-adapted input for one episode."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    episode_id: str
    scene_id: str
    task_name: str
    task_mode: TaskMode
    prompt: str
    goal_metadata: JsonObject = Field(default_factory=dict)
    initial_pose: PoseValues = Field(default_factory=dict)
    output_dir: str = ""


class Observation(RuntimeModel):
    """Serializable observation produced by tools or adapters."""

    summary: str = ""
    image_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    room_id: Optional[str] = None
    pose: PoseValues = Field(default_factory=dict)
    raw: JsonObject = Field(default_factory=dict)


class PlannerDecision(RuntimeModel):
    """Model-selected action after context compilation."""

    action_type: str
    reasoning: str = ""
    expected: str = ""
    confidence: ConfidenceScore = 0.0
    arguments: JsonObject = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_in_range(cls, value: Any) -> Any:
        if value is None:
            return value

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("confidence must be a numeric value")

        value = float(value)
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


class ToolCall(RuntimeModel):
    """Validated tool invocation."""

    call_id: str
    action_type: str
    arguments: JsonObject = Field(default_factory=dict)


class ToolResult(RuntimeModel):
    """Structured result returned by a runtime tool."""

    call_id: str
    action_type: str
    ok: bool
    terminal: bool = False
    observation: Observation = Field(default_factory=Observation)
    error: str = ""
    metrics: MetricsMap = Field(default_factory=dict)


class MemoryPack(RuntimeModel):
    """Context-ready memory query result."""

    query: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: ConfidenceScore = 0.0
    reuse_hint: str = ""


class ContextSection(RuntimeModel):
    """One context section with cache metadata."""

    name: str
    content: str
    cacheable: bool
    token_estimate: NonNegativeInt = 0
    content_hash: str = ""


class EpisodeState(RuntimeModel):
    """Materialized graph state. The event log remains the source of truth."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    episode_id: str
    scene_id: str
    task_name: str
    task_mode: TaskMode
    prompt: str
    round_index: NonNegativeInt = 0
    step_index: NonNegativeInt = 0
    pose: PoseValues = Field(default_factory=dict)
    current_decision: Optional[PlannerDecision] = None
    last_observation: Observation = Field(default_factory=Observation)
    memory_pack: Optional[MemoryPack] = None
    context_sections: list[ContextSection] = Field(default_factory=list)
    terminal: bool = False
    success: bool = False
    answer: str = ""
    failure_type: str = ""
    # Scoring-only fields kept out of prompt context. The context compiler
    # renders only task_instruction fields (episode_id, scene_id, task_name,
    # task_mode, prompt) — these are never part of the rendering.
    distance_to_goal: Optional[NonNegativeFloat] = None
    submitted_explicitly: bool = False


class EpisodeResult(RuntimeModel):
    """Unified output from one episode."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    episode_id: str
    scene_id: str
    task_name: str
    task_mode: TaskMode
    success: bool
    answer: str = ""
    steps_taken: NonNegativeInt = 0
    rounds_used: NonNegativeInt = 0
    path_length: NonNegativeFloat = 0.0
    failure_type: str = ""
    error: str = ""
    event_log_path: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    # GOATBench scoring inputs. `distance_to_goal` is the final
    # agent-to-goal distance (None when not measured, e.g. AEQA episodes).
    # `submit_was_explicit` records whether the terminal state came from an
    # explicit planner submit vs a budget fallback, so the success evaluator
    # can enforce GOATBench's explicit-stop requirement.
    distance_to_goal: Optional[NonNegativeFloat] = None
    submit_was_explicit: bool = False
    # GOATBench: was the target object observed in a snapshot? None when
    # unchecked; the runner sets it from executor last-observation.
    target_observed: Optional[bool] = None


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
    "GoalSpec",
    "BenchmarkRule",
]

PUBLIC_MODELS: dict[str, type[BaseModel]] = {
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
    "GoalSpec": GoalSpec,
    "BenchmarkRule": BenchmarkRule,
}


def dump_runtime_json_schemas() -> dict[str, dict[str, Any]]:
    """Return JSON schemas for public runtime contracts."""

    return {name: model.model_json_schema() for name, model in PUBLIC_MODELS.items()}
