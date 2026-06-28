"""Pydantic contracts for the TierNav runtime."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "tiernav.runtime.v1"
SchemaVersion = Literal["tiernav.runtime.v1"]
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]


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

    schema_version: SchemaVersion = SCHEMA_VERSION
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

    schema_version: SchemaVersion = SCHEMA_VERSION
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
    confidence: ConfidenceScore = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_in_range(cls, value: Any) -> Any:
        if value is None:
            return value

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

    schema_version: SchemaVersion = SCHEMA_VERSION
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

    schema_version: SchemaVersion = SCHEMA_VERSION
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
