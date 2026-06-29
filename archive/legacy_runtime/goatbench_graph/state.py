"""GoatbenchState — LangGraph state schema for the GOATBench subtask loop.

Serializable contract between nodes. Heavy/non-serializable objects (Habitat
scene, TSDF planner, perception models, logger) stay OUT of state — they are
injected via `GoatbenchResources` in `RunnableConfig.configurable`.

Mirrors the variables used by the subtask step loop in
`run_goatbench_evaluation.py` (lines 244-531). Per-step fields use
last-writer-wins (LangGraph default); accumulating lists use `operator.add`.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class GoatbenchState(TypedDict):
    """LangGraph state for one GOATBench subtask.

    Fields grouped by lifecycle:
      - Identity: set once by entrypoint.
      - Budgets: set once, read by nodes/edges.
      - Pose: updated by observe_node / navigate_node.
      - Goal: mutable mapping updated during observe_node.
      - Per-step: overwritten each loop iteration.
      - Accumulating: append-only via `operator.add` reducer.
      - Terminal: set by check_arrival_node.
    """

    # ── Identity (set once by entrypoint) ──
    scene_id: str
    episode_id: str
    subtask_id: str
    question: str
    task_type: str
    goal_class: str
    output_dir: str

    # ── Budgets ──
    max_steps: int
    steps_taken: int

    # ── Pose ──
    pts: Any                    # np.ndarray kept as Any for serde flexibility
    angle: float

    # ── Goal ──
    goal_obj_ids_mapping: dict  # {gt_obj_id: [det_obj_ids]} mutated in observe
    target_obj_ids_estimate: list

    # ── Per-step (last-writer-wins) ──
    rgb_egocentric_views: list
    all_added_obj_ids: list        # newly added objects this step (for update_memory)
    max_point_choice: Any       # SnapShot | Frontier | None
    target_arrived: bool
    n_filtered_snapshots: int

    # ── Accumulating history ──
    step_traces: Annotated[list, operator.add]

    # ── Terminal ──
    task_success: bool
    success_by_snapshot: bool
    success_by_distance: bool
    agent_subtask_distance: float
    n_total_snapshots: int
    n_total_frames: int
    terminal: bool
    error: str
    failure_type: str

    # ── Global step tracking (persists across subtasks) ──
    global_step: int
