"""5 node functions for the GOATBench LangGraph state machine.

Each node is a thin wrapper around the logic in `run_goatbench_evaluation.py`
(lines 244-531). Node signatures use LangGraph's `(state, config) -> partial
state dict` convention. Heavy resources accessed via
`config["configurable"]["resources"]`.

Nodes:
  1. observe_node       — multi-view observation, scene graph update, tsdf integrate
                          (wraps :244-335)
  2. update_memory_node — scene.update_snapshots + tsdf_planner.update_frontier_map
                          (wraps :337-368)
  3. vlm_decide_node    — query_vlm_for_response + set_next_navigation_point
                          (wraps :370-423)
  4. navigate_node      — tsdf_planner.agent_step, update pose, sanity_check, viz
                          (wraps :425-468)
  5. check_arrival_node — check target_arrived, compute success metrics, log result
                          (wraps :470-531; also handles step-budget-exhausted case)
"""
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.habitat import pose_habitat_to_tsdf
from src.tsdf_planner import Frontier, SnapShot
from src.utils import calc_agent_subtask_distance, resize_image
from src.query_vlm_goatbench import query_vlm_for_response

logger = logging.getLogger(__name__)


def observe_node(state, config):
    """Multi-view observation, scene graph update, tsdf integrate.

    Wraps run_goatbench_evaluation.py:244-335.
    """
    res = config["configurable"]["resources"]
    scene = res.scene
    tsdf_planner = res.tsdf_planner
    cfg = res.cfg
    cam_intr = res.cam_intr
    eps_snapshot_dir = res.eps_snapshot_dir
    subtask_metadata = res.subtask_metadata

    pts = state["pts"]
    angle = state["angle"]
    global_step = state["global_step"]
    # Mirror original loop: cnt_step starts at -1, incremented to 0 on first
    # iteration. Entrypoint sets steps_taken=-1; observe increments it.
    cnt_step = state["steps_taken"] + 1
    goal_obj_ids_mapping = state["goal_obj_ids_mapping"]

    global_step += 1
    logging.info(f"\n== step: {cnt_step}, global step: {global_step} ==")

    # (1) Observe surroundings, update scene graph and occupancy map
    # viewing angles
    if cnt_step == 0:
        angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
        total_views = 1 + cfg.extra_view_phase_2
    else:
        angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
        total_views = 1 + cfg.extra_view_phase_1
    all_angles = [
        angle + angle_increment * (i - total_views // 2) for i in range(total_views)
    ]
    # Let the main viewing angle be the last one to avoid potential overwriting problems
    main_angle = all_angles.pop(total_views // 2)
    all_angles.append(main_angle)

    rgb_egocentric_views = []
    all_added_obj_ids = set()

    # Record all the objects newly added in this step
    for view_idx, ang in enumerate(all_angles):
        # each view
        obs, cam_pose = scene.get_observation(pts, angle=ang)
        rgb = obs["color_sensor"]
        depth = obs["depth_sensor"]
        semantic_obs = obs["semantic_sensor"]

        # all view features
        obs_file_name = f"{global_step}-view_{view_idx}.png"

        with torch.no_grad():
            # Concept graph pipeline update
            annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                scene.update_scene_graph(
                    image_rgb=rgb[..., :3],
                    depth=depth,
                    intrinsics=cam_intr,
                    cam_pos=cam_pose,
                    pts=pts,
                    pts_voxel=tsdf_planner.habitat2voxel(pts),
                    img_path=obs_file_name,
                    frame_idx=cnt_step * total_views + view_idx,
                    semantic_obs=semantic_obs,
                    gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                )
            )

        scene.all_observations[obs_file_name] = rgb
        rgb_egocentric_views.append(
            resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
        )

        if cfg.save_visualization:
            plt.imsave(
                os.path.join(eps_snapshot_dir, obs_file_name), annotated_rgb
            )
        else:
            plt.imsave(
                os.path.join(eps_snapshot_dir, obs_file_name), rgb
            )

        # update the mapping from hm3d object id to our detected object id
        for gt_goal_id, det_goal_id in target_obj_id_mapping.items():
            goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
        all_added_obj_ids.update(added_obj_ids)

        # Clean up redundant objects periodically
        scene.periodic_cleanup_objects(
            frame_idx=cnt_step * total_views + view_idx,
            pts=pts,
            goal_obj_ids_mapping=goal_obj_ids_mapping,
        )

        # Update depth map, occupancy map
        tsdf_planner.integrate(
            color_im=rgb,
            depth_im=depth,
            cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose),
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )
    logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")

    return {
        "pts": pts,
        "angle": angle,
        "rgb_egocentric_views": rgb_egocentric_views,
        "all_added_obj_ids": list(all_added_obj_ids),
        "goal_obj_ids_mapping": goal_obj_ids_mapping,
        "global_step": global_step,
        "steps_taken": cnt_step,
    }


def update_memory_node(state, config):
    """scene.update_snapshots + tsdf_planner.update_frontier_map.

    Wraps run_goatbench_evaluation.py:337-368. Side effects on scene/tsdf_planner.
    """
    res = config["configurable"]["resources"]
    scene = res.scene
    tsdf_planner = res.tsdf_planner
    cfg = res.cfg
    eps_frontier_dir = res.eps_frontier_dir
    cnt_step = state["steps_taken"]
    pts = state["pts"]

    # (2) Update Memory Snapshots with hierarchical clustering
    # Choose all the newly added objects as well as the objects nearby as the cluster targets
    all_added_obj_ids = [
        obj_id
        for obj_id in state["all_added_obj_ids"]
        if obj_id in scene.objects
    ]
    for obj_id, obj in scene.objects.items():
        if (
            np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
            < cfg.scene_graph.obj_include_dist + 0.5
        ):
            if obj_id not in all_added_obj_ids:
                all_added_obj_ids.append(obj_id)

    scene.update_snapshots(
        obj_ids=set(all_added_obj_ids),
        min_detection=cfg.min_detection,
    )
    logging.info(
        f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
    )

    # (3) Update Frontier Snapshots
    update_success = tsdf_planner.update_frontier_map(
        pts=pts,
        cfg=cfg.planner,
        scene=scene,
        cnt_step=cnt_step,
        save_frontier_image=cfg.save_visualization,
        eps_frontier_dir=eps_frontier_dir,
        prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
    )
    if not update_success:
        logging.info("Warning! Update frontier map failed!")

    return {}


def vlm_decide_node(state, config):
    """query_vlm_for_response + set_next_navigation_point.

    Wraps run_goatbench_evaluation.py:370-423.
    Returns terminal=True if VLM or set_next_navigation_point fails.
    """
    res = config["configurable"]["resources"]
    scene = res.scene
    tsdf_planner = res.tsdf_planner
    cfg = res.cfg
    subtask_metadata = res.subtask_metadata
    subtask_id = state["subtask_id"]

    pts = state["pts"]
    rgb_egocentric_views = state["rgb_egocentric_views"]
    goal_obj_ids_mapping = state["goal_obj_ids_mapping"]

    # (4) point querying VLM
    if cfg.choose_every_step:
        # choose and query vlm at every step, clear the target point
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None
    else:
        # if already has target point, allow the model to choose again
        if tsdf_planner.max_point is not None:
            if type(tsdf_planner.max_point) == Frontier:
                # the target is a frontier point, allow the model to choose again
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None

    # use the most common id in the mapped ids as the detected target object id
    target_obj_ids_estimate = []
    for obj_id, det_ids in goal_obj_ids_mapping.items():
        if len(det_ids) == 0:
            continue
        target_obj_ids_estimate.append(
            max(set(det_ids), key=det_ids.count)
        )

    tsdf_planner.max_point = None
    tsdf_planner.target_point = None

    # query VLM for next navigation point, and reason the choice
    vlm_response = query_vlm_for_response(
        subtask_metadata=subtask_metadata,
        scene=scene,
        tsdf_planner=tsdf_planner,
        rgb_egocentric_views=rgb_egocentric_views,
        cfg=cfg,
        verbose=True,
    )

    if vlm_response is None:
        logging.info(
            f"Subtask id {subtask_id} invalid: query_vlm_for_response failed!"
        )
        return {
            "max_point_choice": None,
            "n_filtered_snapshots": 0,
            "target_obj_ids_estimate": target_obj_ids_estimate,
            "error": "query_vlm_for_response failed",
            "failure_type": "vlm_failed",
            "terminal": True,
        }

    max_point_choice, n_filtered_snapshots = vlm_response

    # set the vlm choice as the navigation target
    update_success = tsdf_planner.set_next_navigation_point(
        choice=max_point_choice,
        pts=pts,
        objects=scene.objects,
        cfg=cfg.planner,
        pathfinder=scene.pathfinder,
    )

    if not update_success:
        logging.info(
            f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!"
        )
        return {
            "max_point_choice": max_point_choice,
            "n_filtered_snapshots": n_filtered_snapshots,
            "target_obj_ids_estimate": target_obj_ids_estimate,
            "error": "set_next_navigation_point failed",
            "failure_type": "set_next_navigation_point_failed",
            "terminal": True,
        }

    return {
        "max_point_choice": max_point_choice,
        "n_filtered_snapshots": n_filtered_snapshots,
        "target_obj_ids_estimate": target_obj_ids_estimate,
        "error": "",
        "failure_type": "",
        "terminal": False,
    }


def navigate_node(state, config):
    """tsdf_planner.agent_step, update pose, sanity_check, viz.

    Wraps run_goatbench_evaluation.py:425-468.
    Returns terminal=True if agent_step fails.
    """
    res = config["configurable"]["resources"]
    scene = res.scene
    tsdf_planner = res.tsdf_planner
    cfg = res.cfg
    run_logger = res.logger
    eps_frontier_dir = res.eps_frontier_dir
    subtask_metadata = res.subtask_metadata

    pts = state["pts"]
    angle = state["angle"]
    global_step = state["global_step"]
    subtask_id = state["subtask_id"]
    max_point_choice = state["max_point_choice"]
    goal_obj_ids_mapping = state["goal_obj_ids_mapping"]

    # (5) Agent navigate to the target point for one step
    return_values = tsdf_planner.agent_step(
        pts=pts,
        angle=angle,
        objects=scene.objects,
        snapshots=scene.snapshots,
        pathfinder=scene.pathfinder,
        cfg=cfg.planner,
        path_points=None,
        save_visualization=cfg.save_visualization,
    )
    if return_values[0] is None:
        logging.info(
            f"Subtask id {subtask_id} invalid: agent_step failed!"
        )
        return {
            "pts": pts,
            "angle": angle,
            "target_arrived": False,
            "error": "agent_step failed",
            "failure_type": "agent_step_failed",
            "terminal": True,
        }

    # update agent's position and rotation
    pts, angle, pts_voxel, fig, _, target_arrived = return_values
    run_logger.log_step(pts_voxel=pts_voxel)
    logging.info(
        f"Current position: {pts}, {run_logger.subtask_explore_dist:.3f}"
    )

    # sanity check about objects, scene graph, snapshots, ...
    scene.sanity_check(cfg=cfg)

    if cfg.save_visualization:
        # save the top-down visualization
        run_logger.save_topdown_visualization(
            global_step=global_step,
            subtask_id=subtask_id,
            subtask_metadata=subtask_metadata,
            goal_obj_ids_mapping=goal_obj_ids_mapping,
            fig=fig,
        )
        # save the visualization of vlm's choice at each step
        run_logger.save_frontier_visualization(
            global_step=global_step,
            subtask_id=subtask_id,
            tsdf_planner=tsdf_planner,
            max_point_choice=max_point_choice,
            global_caption=f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}",
        )

    return {
        "pts": pts,
        "angle": angle,
        "target_arrived": target_arrived,
        "error": "",
        "failure_type": "",
        "terminal": False,
    }


def check_arrival_node(state, config):
    """Check target_arrived, compute success metrics, log subtask result.

    Wraps run_goatbench_evaluation.py:470-531. Handles three cases:
      1. target_arrived + SnapShot choice → success path: save snapshot, score, terminal.
      2. Step budget exhausted (steps_taken >= max_steps - 1) → score with
         task_success=False, terminal.
      3. Otherwise → terminal=False, loop back to observe.

    Mirrors the original: the `if type(max_point_choice)==SnapShot and target_arrived`
    block (lines 470-489) runs inside the loop; the scoring block (491-531) runs
    after the loop exits (via break or budget exhaustion).
    """
    res = config["configurable"]["resources"]
    scene = res.scene
    tsdf_planner = res.tsdf_planner
    cfg = res.cfg
    run_logger = res.logger
    eps_snapshot_dir = res.eps_snapshot_dir
    subtask_metadata = res.subtask_metadata

    pts = state["pts"]
    angle = state["angle"]
    max_point_choice = state.get("max_point_choice")
    target_arrived = state.get("target_arrived", False)
    n_filtered_snapshots = state.get("n_filtered_snapshots", 0)
    target_obj_ids_estimate = state.get("target_obj_ids_estimate", [])
    goal_type = state["task_type"]  # goal_type mirrors task_type in original
    subtask_id = state["subtask_id"]
    steps_taken = state["steps_taken"]
    max_steps = state["max_steps"]

    task_success = False

    # If a prior node (vlm_decide / navigate) already marked terminal due to
    # failure, skip the arrival/budget logic and go straight to scoring with
    # task_success=False (mirrors original `break` falling through to scoring).
    if state.get("terminal", False):
        # max_point_choice may be None here; guard the snapshot check below.
        pass
    # (6) Check if the agent has arrived at the target to finish the question
    elif type(max_point_choice) == SnapShot and target_arrived:
        # when the target is a snapshot, and the agent arrives at the target
        # we consider the subtask is finished, take an observation and save the chosen target snapshot
        obs, _ = scene.get_observation(pts, angle=angle)
        rgb = obs["color_sensor"]
        plt.imsave(
            os.path.join(
                run_logger.subtask_object_observe_dir, f"target.png"
            ),
            rgb,
        )

        snapshot_filename = max_point_choice.image.split(".")[0]
        os.system(
            f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(run_logger.subtask_object_observe_dir, f'snapshot_{snapshot_filename}.png')}"
        )

        task_success = True
        # fall through to scoring + terminal
    elif steps_taken < max_steps - 1:
        # budget remains — loop back for another step
        return {
            "task_success": False,
            "terminal": False,
        }
    # else: step budget exhausted — fall through to scoring with task_success=False

    # ── Scoring (mirrors :491-531; runs on success break OR budget exhaustion) ──
    if task_success and np.any(
        [
            obj_id in max_point_choice.cluster
            for obj_id in target_obj_ids_estimate
        ]
    ):
        success_by_snapshot = True
        logging.info(
            f"Success: {target_obj_ids_estimate} in chosen snapshot {max_point_choice.image}!"
        )
    else:
        success_by_snapshot = False
        logging.info(
            f"Fail: {target_obj_ids_estimate} not in chosen snapshot!"
        )
    # calculate the distance to the nearest view point
    agent_subtask_distance = calc_agent_subtask_distance(
        pts, subtask_metadata["viewpoints"], scene.pathfinder
    )
    if agent_subtask_distance < cfg.success_distance:
        success_by_distance = True
        logging.info(
            f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!"
        )
    else:
        success_by_distance = False
        logging.info(
            f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!"
        )

    run_logger.log_subtask_result(
        success_by_snapshot=success_by_snapshot,
        success_by_distance=success_by_distance,
        subtask_id=subtask_id,
        gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
        goal_type=goal_type,
        n_filtered_snapshots=n_filtered_snapshots,
        n_total_snapshots=len(scene.snapshots),
        n_total_frames=len(scene.frames),
    )

    # Per-subtask checkpoint — persist progress even if next subtask crashes
    run_logger.save_results()

    logging.info(f"Scene graph of question {subtask_id}:")
    logging.info(f"Question: {subtask_metadata['question']}")
    logging.info(f"Task type: {subtask_metadata['task_type']}")
    logging.info(f"Answer: {subtask_metadata['class']}")
    scene.print_scene_graph()

    if not cfg.save_visualization:
        # clear up the stored images to save memory
        os.system(
            f"rm -r {os.path.join(str(cfg.output_dir), f'{subtask_id}')}"
        )

    return {
        "task_success": task_success,
        "success_by_snapshot": success_by_snapshot,
        "success_by_distance": success_by_distance,
        "agent_subtask_distance": agent_subtask_distance,
        "n_total_snapshots": len(scene.snapshots),
        "n_total_frames": len(scene.frames),
        "terminal": True,
    }
