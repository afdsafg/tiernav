import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)
os.environ.setdefault("MAGNUM_GPU_CONTEXT", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "0")

import argparse
import math
import json
import logging
import time
import random

import numpy as np
import torch
from omegaconf import OmegaConf

import open_clip
from ultralytics import SAM, YOLOWorld

from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, SnapShot
from src.scene_goatbench import Scene
from src.utils import calc_agent_subtask_distance, get_pts_angle_goatbench
from src.goatbench_utils import prepare_goatbench_navigation_goals
from src.logger_goatbench import Logger
from src.two_tier_graph.entrypoint import run_episode_two_tier_langgraph

CORRUPTED_SCENES: set[str] = set()


def _corrupted_scenes_path(output_dir: str) -> str:
    return os.path.join(output_dir, "corrupted_scenes.json")


def _save_corrupted_scenes(scenes: set, output_dir: str):
    path = _corrupted_scenes_path(output_dir)
    with open(path, "w") as f:
        json.dump(sorted(scenes), f, indent=2)


def _load_corrupted_scenes(output_dir: str) -> set:
    path = _corrupted_scenes_path(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def _build_goal_question(goal_type: str, subtask_goal: list, goal_category: str) -> str:
    """Build the navigation question for the planner."""
    if goal_type == "object":
        return f"Navigate to find the {goal_category}."
    elif goal_type == "description":
        return f"Navigate to find the object exactly described as: '{subtask_goal[0]['lang_desc']}'."
    else:
        return "Navigate to find the exact object captured in the target image. Pay attention to the environment to find the exact object."


def _check_snapshot_success(scene, subtask_metadata, goal_obj_ids_mapping) -> bool:
    """Check if any goal object is in any snapshot cluster."""
    if not hasattr(scene, "snapshots") or not scene.snapshots:
        return False
    goal_obj_ids = subtask_metadata["goal_obj_ids"]
    for snap in scene.snapshots:
        if isinstance(snap, SnapShot):
            cluster = snap.cluster
        elif isinstance(snap, dict):
            cluster = snap.get("cluster", [])
        else:
            continue
        for det_id in goal_obj_ids_mapping.get(goal_obj_ids[0], []):
            if det_id in cluster:
                return True
        for gid in goal_obj_ids:
            for det_id in goal_obj_ids_mapping.get(gid, []):
                if det_id in cluster:
                    return True
    return False


def run_goatbench_two_tier(
    scene,
    tsdf_planner,
    cfg,
    cam_intr,
    logger: Logger,
    models: dict,
    episode_dir: str,
    eps_snapshot_dir: str,
    scene_id: str,
    episode_id: str,
    all_subtask_goal_types: list,
    all_subtask_goals: list,
    pts,
    angle,
    num_step: int,
    cfg_cg,
    detection_model,
    sam_predictor,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
):
    """Run GOATBench episode using Two-Tier Planner-Executor workflow."""
    global_step = -1

    # 跨 subtask 线程资源（GOATBench 允许跨 subtask 记忆）
    threaded_notebook = None
    threaded_scene_graph = None
    cross_subtask_notes = []

    for subtask_idx, (goal_type, subtask_goal) in enumerate(
        zip(all_subtask_goal_types, all_subtask_goals)
    ):
        subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
        logging.info(
            f"\n=== Scene {scene_id} Episode {episode_id} "
            f"Subtask {subtask_idx + 1}/{len(all_subtask_goals)} ==="
        )

        # init_subtask needs scene.pathfinder + tsdf_planner.habitat2voxel
        subtask_metadata = logger.init_subtask(
            subtask_id=subtask_id,
            goal_type=goal_type,
            subtask_goal=subtask_goal,
            pts=pts,
            scene=scene,
            tsdf_planner=tsdf_planner,
        )

        # Build goal→det mapping (will be populated during exploration)
        goal_obj_ids_mapping = {
            obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
        }

        # Build navigation question for the planner
        goal_question = _build_goal_question(
            goal_type, subtask_goal, subtask_metadata["class"]
        )

        # Reset tsdf planner for new subtask
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None

        # Clear memory between subtasks if configured
        if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
            scene.clear_up_detections()

        workflow_output_dir = os.path.join(
            cfg.output_dir, "two_tier_workflow", subtask_id
        )

        try:
            # goal_description: 仅人类可读描述，绝不含真值坐标
            if goal_type == "object":
                goal_desc = subtask_metadata.get("class", "")
            elif goal_type == "description" and isinstance(subtask_goal, list) and len(subtask_goal) > 0:
                goal_desc = subtask_goal[0].get("lang_desc", "") if isinstance(subtask_goal[0], dict) else str(subtask_goal[0])
            elif goal_type == "description":
                goal_desc = str(subtask_goal)
            else:
                goal_desc = ""  # image goal: no text description
            goal_metadata = {
                "goal_description": goal_desc,
                "goal_type": goal_type,
            }

            result = run_episode_two_tier_langgraph(
                scene_id=scene_id,
                question=goal_question,
                question_id=subtask_id,
                cfg=cfg,
                detection_model=detection_model,
                sam_predictor=sam_predictor,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
                output_dir=workflow_output_dir,
                max_planner_rounds=num_step,
                max_total_steps=num_step,
                start_pts=pts,
                start_angle=angle,
                run_logger=None,
                method_config={
                    "use_notebook": True,
                    "use_scene_graph": True,
                    "use_active_query": True,
                    "use_rejected_tracking": True,
                },
                scene=scene,
                tsdf_planner=tsdf_planner,
                notebook=threaded_notebook,
                scene_graph=threaded_scene_graph,
                goal_type=goal_type,
                goal_metadata=goal_metadata,
                subtask_index=subtask_idx,
                subtask_total=len(all_subtask_goals),
                cross_subtask_notes=cross_subtask_notes,
            )
        except Exception as e:
            logging.exception(f"Subtask {subtask_id} failed: {e}")
            result = {
                "scene_id": scene_id,
                "question_id": subtask_id,
                "question": goal_question,
                "answer": "",
                "success": False,
                "steps_taken": 0,
                "path_length": 0.0,
                "rounds_used": 0,
                "error": str(e),
                "final_pts": pts,
                "final_angle": angle,
                "_notebook": threaded_notebook,
                "_scene_graph": threaded_scene_graph,
                "cross_subtask_notes": cross_subtask_notes,
            }

        # 回收线程资源供下一 subtask
        threaded_notebook = result.get("_notebook", threaded_notebook)
        threaded_scene_graph = result.get("_scene_graph", threaded_scene_graph)
        cross_subtask_notes = result.get("cross_subtask_notes", cross_subtask_notes)

        # Thread pose across subtasks
        final_pts = result.get("final_pts", pts)
        final_angle = result.get("final_angle", angle)
        if final_pts is not None and not np.isnan(final_pts).any():
            pts = np.asarray(final_pts)
            angle = final_angle

        # ── Score ──────────────────────────────────────────────────
        # Update goal_obj_ids_mapping from scene objects
        for obj_id, obj in scene.objects.items():
            if not isinstance(obj, dict) or "class_name" not in obj:
                continue
            for gt_id in goal_obj_ids_mapping:
                if str(gt_id) in str(obj.get("instance_id", "")):
                    if obj_id not in goal_obj_ids_mapping[gt_id]:
                        goal_obj_ids_mapping[gt_id].append(obj_id)

        # Snapshot success
        success_by_snapshot = _check_snapshot_success(
            scene, subtask_metadata, goal_obj_ids_mapping
        )

        # Distance success
        agent_subtask_distance = calc_agent_subtask_distance(
            pts, subtask_metadata["viewpoints"], scene.pathfinder
        )
        success_by_distance = agent_subtask_distance < cfg.success_distance

        logging.info(
            f"Subtask {subtask_id}: snapshot={success_by_snapshot}, "
            f"distance={success_by_distance} ({agent_subtask_distance:.2f}m)"
        )

        # Log results
        logger.log_subtask_result(
            success_by_snapshot=success_by_snapshot,
            success_by_distance=success_by_distance,
            subtask_id=subtask_id,
            gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
            goal_type=goal_type,
            n_filtered_snapshots=0,
            n_total_snapshots=len(scene.snapshots) if hasattr(scene, "snapshots") else 0,
            n_total_frames=result.get("steps_taken", 0),
        )

        # Per-subtask checkpoint
        logger.save_results()

        logging.info(f"Subtask {subtask_id} finished:")
        logging.info(f"  Goal: {goal_question}")
        logging.info(f"  Steps: {result.get('steps_taken', 0)}")
        logging.info(f"  Rounds: {result.get('rounds_used', 0)}")
        if result.get("error"):
            logging.warning(f"  Error: {result['error']}")

    return global_step


def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1):
    global CORRUPTED_SCENES
    CORRUPTED_SCENES = _load_corrupted_scenes(cfg.output_dir)

    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)

    scene_data_list = scene_data_list[
        int(start_ratio * num_scene) : int(end_ratio * num_scene)
    ]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
            num_episode += len(json.load(f)["episodes"])
    logging.info(
        f"Total number of episodes: {num_episode}; Selected episodes: {len(scene_data_list)}"
    )
    logging.info(f"Total number of scenes: {len(scene_data_list)}")

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )

    # Load models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )

    for scene_data_file in scene_data_list:
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(
            open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
        )

        scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
        total_episodes = len(scene_data["episodes"])

        all_navigation_goals = scene_data["goals"]

        if scene_name in CORRUPTED_SCENES:
            logging.warning(f"Skipping known-corrupted scene: {scene_name}")
            continue

        for episode_idx, episode in enumerate(scene_data["episodes"]):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            all_subtask_goal_types, all_subtask_goals = (
                prepare_goatbench_navigation_goals(
                    scene_name=scene_name,
                    episode=episode,
                    all_navigation_goals=all_navigation_goals,
                )
            )

            # Skip already-processed episodes
            finished_subtask_ids = list(logger.success_by_snapshot.keys())
            finished_episode_subtask = [
                sid
                for sid in finished_subtask_ids
                if sid.startswith(f"{scene_id}_{episode_id}_")
            ]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # Load scene
            try:
                del scene
            except:
                pass
            try:
                scene = Scene(
                    scene_id,
                    cfg,
                    cfg_cg,
                    detection_model,
                    sam_predictor,
                    clip_model,
                    clip_preprocess,
                    clip_tokenizer,
                )

                floor_height = pts[1]
                tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
                num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
                num_step = max(num_step, 50)
                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds,
                    voxel_size=cfg.tsdf_grid_size,
                    floor_height=floor_height,
                    floor_height_offset=0,
                    pts_init=pts,
                    init_clearance=cfg.init_clearance * 2,
                    save_visualization=cfg.save_visualization,
                )

                episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                    episode_id=f"{scene_id}_ep_{episode_id}"
                )

                logging.info(f"\n\nScene {scene_id} initialization successful!")

                models = {
                    "detection": detection_model,
                    "sam": sam_predictor,
                    "clip": clip_model,
                    "clip_preprocess": clip_preprocess,
                    "clip_tokenizer": clip_tokenizer,
                }

                run_goatbench_two_tier(
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    cfg=cfg,
                    cam_intr=cam_intr,
                    logger=logger,
                    models=models,
                    episode_dir=episode_dir,
                    eps_snapshot_dir=eps_snapshot_dir,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    all_subtask_goal_types=all_subtask_goal_types,
                    all_subtask_goals=all_subtask_goals,
                    pts=pts,
                    angle=angle,
                    num_step=num_step,
                    cfg_cg=cfg_cg,
                    detection_model=detection_model,
                    sam_predictor=sam_predictor,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    clip_tokenizer=clip_tokenizer,
                )

                logger.save_results()
                logging.info(f"Episode {episode_id} finish")

            except Exception as e:
                logging.error(
                    f"Scene {scene_id} crashed: {e}. Marking as corrupted and skipping."
                )
                import traceback
                traceback.print_exc()
                CORRUPTED_SCENES.add(scene_name)
                _save_corrupted_scenes(CORRUPTED_SCENES, cfg.output_dir)
                try:
                    logger.save_results()
                except Exception:
                    pass
                continue

    logger.save_results()
    logger.aggregate_results()
    logging.info(f"All scenes finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    logging.info(f"***** Running {cfg.exp_name} (Two-Tier) *****")
    main(
        cfg,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
        split=args.split,
    )
