"""TwoTierState — LangGraph state schema for the Two-Tier loop.

Serializable contract between nodes. Heavy/non-serializable objects (perception
models, Habitat scene, TSDF volumetric map, planner/executor instances) stay OUT
of state — they are injected via `Resources` in `RunnableConfig.configurable`.

The schema mirrors the variables used by `run_episode_two_tier`
(agent_workflow.py:1087-1702). Per-round fields use last-writer-wins (LangGraph
default); accumulating lists use `operator.add` so future parallel/multi-agent
writes merge correctly.
"""
from __future__ import annotations

import enum
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict


class TransitionReason(str, enum.Enum):
    CONTINUE = "continue"
    ROUND_BUDGET = "round_budget"
    EXHAUSTED = "exhausted"
    STEP_BUDGET = "step_budget"
    STALL_RECOVERY = "stall_recovery"  # P3
    VERIFY_BEFORE_FALLBACK = "verify_before_fallback"  # P3


@dataclass
class Transition:
    reason: TransitionReason
    from_node: str
    to_node: str
    round_idx: int


class CurrentPose(TypedDict):
    """Agent pose — mirrors Executor._pts / Executor._angle."""

    pts: Optional[Any]        # np.ndarray kept as Any for serde flexibility
    angle: float


class TwoTierState(TypedDict):
    """LangGraph state for the Two-Tier Planner-Executor loop.

    Fields are grouped by lifecycle:
      - Episode identity / budgets / method flags: set once by `init_node`.
      - Mutable per-round: updated each round by planner/executor/memory nodes.
      - Accumulating history: append-only via `operator.add` reducer.
      - Per-round prompt artifacts: overwritten each `build_context_node` round.
      - Terminal: set by `submit_node`.
    """

    # ── Episode identity (set once by init_node) ──
    scene_id: str
    question_id: str
    question: str
    output_dir: str

    # ── Budgets (set once by init_node, read by edges) ──
    max_planner_rounds: int
    max_total_steps: int

    # ── Method / ablation flags (set once) ──
    use_notebook: bool
    use_scene_graph: bool
    use_active_query: bool
    use_rejected_tracking: bool

    # ── Mutable agent state (per-round) ──
    pose: CurrentPose
    rounds_used: int
    steps_taken: int
    current_action: Optional[Any]            # PlannerAction dataclass instance
    last_evidence: Optional[Any]             # TrajectoryEvidence dataclass instance
    exhausted_flag: bool                     # notebook.is_exhausted(seed_id) result

    # ── Accumulating history (append reducers) ──
    # action_history holds action_type strings; round_traces holds RoundTrace
    # dataclass instances (agent_workflow.RoundTrace) — kept as Any to avoid
    # heavy imports in this module.
    action_history: Annotated[list, operator.add]
    round_traces: Annotated[list, operator.add]

    # ── Compression config + log (P0a: layered compression) ──
    compress_threshold: int                  # L_compressed trigger, default 5
    index_refresh_interval: int              # L_index trigger, default 3
    l0_index_text: str                       # cached L0 index string (always-in-prompt)
    visual_memory_state: dict                # serialized VisualMemoryIndex (entries/text/round)
    loaded_snapshot_ids: list                # snapshot_ids seen this episode (cross-round dedup)
    compression_log: Annotated[list, operator.add]  # per-layer stats

    # ── Transition tracking (P0b: transition.reason) ──
    last_transition: Optional[dict]               # Transition as dict (serializable)
    transition_log: Annotated[list, operator.add]  # accumulates per-round

    # ── Stall detection (P3) ──
    stall_signal: Optional[dict]               # serialized StallSignal
    verification_attempted: bool               # P3 verify nudge flag

    # ── Per-round prompt/context artifacts (last-writer-wins) ──
    scene_analysis: str
    history_text: str
    progress_text: str
    actions_text: str
    current_views: list                      # list[dict] with snapshot_id/view_idx/direction/image_b64
    topdown_b64: Optional[str]
    memory_summary: dict                     # from scene_graph active query

    # ── Terminal ──
    answer: str
    success: bool
    error: str
    terminal: bool
    failure_type: str                        # "budget_exhausted_answered" | "premature_submit" | ...
