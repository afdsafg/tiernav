"""run_goatbench_subtask_langgraph — LangGraph entrypoint for GOATBench subtasks.

Same signature as `run_goatbench_subtask_legacy`. Iterates over all subtasks in
the episode; for each subtask: performs per-subtask setup (lines 199-243 of
run_goatbench_evaluation.py), invokes the compiled graph for the step loop
(lines 244-489), and lets check_arrival_node handle post-loop scoring
(lines 491-531).

Heavy objects (scene, tsdf_planner, models, logger) go in GoatbenchResources,
not State. The subtask_metadata from logger.init_subtask() goes in Resources
(read-only handle). goal_obj_ids_mapping is in State (mutable dict, updated
during observe_node).

Returns updated global_step (threaded across subtasks), matching legacy.
"""
from __future__ import annotations

import logging

from src.tsdf_planner import TSDFPlanner

from .graph import build_goatbench_graph
from .resources import GoatbenchResources
from .state import GoatbenchState

logger = logging.getLogger(__name__)


def run_goatbench_subtask_langgraph(
    scene,
    tsdf_planner,
    cfg,
    cam_intr,
    logger,  # src.logger_goatbench.Logger
    models: dict,
    eps_frontier_dir: str,
    eps_snapshot_dir: str,
    episode_dir: str,
    scene_id: str,
    episode_id: str,
    all_subtask_goal_types,
    all_subtask_goals,
    pts,
    angle,
    global_step,
    num_step,
    cfg_cg,
    tsdf_bnds,
    floor_height: float,
):
    """Run all GOATBench subtasks for one episode via LangGraph.

    Returns the updated `global_step` (mirrors legacy behavior where the caller
    threads global_step across subtasks).
    """
    run_logger = logger
    graph = build_goatbench_graph()

    for subtask_idx, (goal_type, subtask_goal) in enumerate(
        zip(all_subtask_goal_types, all_subtask_goals)
    ):
        subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
        logging.info(
            f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}"
        )

        # ── Per-subtask setup (mirrors :207-243) ──
        subtask_metadata = run_logger.init_subtask(
            subtask_id=subtask_id,
            goal_type=goal_type,
            subtask_goal=subtask_goal,
            pts=pts,
            scene=scene,
            tsdf_planner=tsdf_planner,
        )

        # mapping from the obj id in habitat to the id assigned by concept graph
        goal_obj_ids_mapping = {
            obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
        }

        # reset tsdf planner
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None

        if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
            scene.clear_up_detections()
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

        # ── Build Resources ──
        resources = GoatbenchResources(
            scene=scene,
            tsdf_planner=tsdf_planner,
            cfg=cfg,
            cam_intr=cam_intr,
            logger=run_logger,
            models=models,
            eps_frontier_dir=eps_frontier_dir,
            eps_snapshot_dir=eps_snapshot_dir,
            episode_dir=episode_dir,
            subtask_metadata=subtask_metadata,
        )

        # ── Build initial state ──
        # steps_taken mirrors cnt_step=-1 (observe_node increments to 0 on first iter).
        initial_state: GoatbenchState = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "subtask_id": subtask_id,
            "question": subtask_metadata.get("question", ""),
            "task_type": goal_type,
            "goal_class": subtask_metadata.get("class", ""),
            "output_dir": cfg.output_dir,
            "max_steps": num_step,
            "steps_taken": -1,
            "pts": pts,
            "angle": angle,
            "goal_obj_ids_mapping": goal_obj_ids_mapping,
            "target_obj_ids_estimate": [],
            "rgb_egocentric_views": [],
            "all_added_obj_ids": [],
            "max_point_choice": None,
            "target_arrived": False,
            "n_filtered_snapshots": 0,
            "step_traces": [],
            "task_success": False,
            "success_by_snapshot": False,
            "success_by_distance": False,
            "agent_subtask_distance": 0.0,
            "n_total_snapshots": 0,
            "n_total_frames": 0,
            "terminal": False,
            "error": "",
            "failure_type": "",
            "global_step": global_step,
        }

        # ── Invoke graph ──
        # recursion_limit: 5 nodes per step iteration (observe→update_memory→vlm_decide
        # →navigate→check_arrival) plus headroom for the final terminal pass.
        final_state = graph.invoke(
            initial_state,
            config={
                "configurable": {"resources": resources},
                "recursion_limit": num_step * 5 + 20,
            },
        )

        # Thread pts/angle/global_step across subtasks (updated by navigate_node).
        pts = final_state.get("pts", pts)
        angle = final_state.get("angle", angle)
        global_step = final_state.get("global_step", global_step)

    return global_step
