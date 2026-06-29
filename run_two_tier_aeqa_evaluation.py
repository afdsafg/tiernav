import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# EGL headless rendering (required for NVIDIA GPU without display)
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)
os.environ.setdefault("MAGNUM_GPU_CONTEXT", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "0")

import argparse
import functools
import json
import logging
import multiprocessing as mp
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from omegaconf import OmegaConf
from ultralytics import SAM, YOLOWorld

from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.contracts import PlannerDecision
from src.const import INVALID_SCENE_ID

# Runtime engine — only path after Task 9.  _run_aeqa_runtime uses the
# TierNav runtime with adapters and RuntimeEntrypoint (Task 8).  Legacy and
# LangGraph paths were removed in Task 9 (commit 4951b10+1).


def _run_aeqa_runtime(
    scene_id,
    question,
    question_id,
    cfg,
    detection_model,
    sam_predictor,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    output_dir,
    max_planner_rounds,
    max_total_steps,
    start_pts,
    start_angle,
    run_logger,
    method_config,
    scene_class=None,
    goal_type=None,
    goal_metadata=None,
    scene=None,
    tsdf_planner=None,
):
    """Runtime engine: adapter -> RuntimeEntrypoint -> legacy dict.

    Structural integration path introduced in Task 8.  Uses the AEQA task
    adapter and RuntimeEntrypoint.with_fake_services (stable defaults) to
    produce a correctly-shaped result dict.  Task 9 replaces the fake
    services with habitat-backed tools and a real VLM planner.
    """
    from src.tiernav_runtime.adapters import AEQATaskAdapter
    from src.tiernav_runtime.contracts import RunSpec
    from src.tiernav_runtime.entrypoint import (
        RuntimeEntrypoint,
        episode_result_to_legacy_dict,
    )

    adapter = AEQATaskAdapter()

    provider_config = ProviderConfig(
        api_key_env="QWEN_PLANNER_API_KEY",
        base_url_env="QWEN_PLANNER_BASE_URL",
        model_env="QWEN_PLANNER_MODEL",
    )

    pts_array = getattr(start_pts, "tolist", None)
    if pts_array is not None:
        pts_list = pts_array()
    else:
        pts_list = list(start_pts) if start_pts is not None else [0.0, 0.0]
    initial_pose = {
        "x": float(pts_list[0]),
        "y": float(pts_list[1]),
        "theta": float(start_angle),
    }

    request = adapter.to_request(
        scene_id=scene_id,
        question_id=question_id,
        question=question,
        output_dir=output_dir,
        initial_pose=initial_pose,
    )

    spec = RunSpec(
        run_id=question_id,
        task_name="aeqa",
        dataset_split="aeqa",
        output_dir=output_dir,
        planner_provider=provider_config.resolve_base_url(),
        planner_model=provider_config.resolve_model(),
        max_rounds=max_planner_rounds,
        max_steps=max_total_steps,
    )

    # Fake planner: submits an empty answer.  The result dict has the
    # correct shape but not a real VLM-generated answer.  Task 9 will
    # replace this with a PlannerClient backed by the real VLM.
    _fake_planner = type(
        "FakePlanner",
        (),
        {
            "decide": lambda self, prompt: PlannerDecision(
                action_type="submit_answer", arguments={"answer": ""}
            )
        },
    )()

    entrypoint = RuntimeEntrypoint.with_fake_services(_fake_planner)
    result = entrypoint.run(spec, request)
    return episode_result_to_legacy_dict(result, question=question)


from src.logger_aeqa import Logger
from src.utils import get_pts_angle_aeqa

# Monkey-patch torch.load for PyTorch 2.6 compatibility
# (2.6 changed weights_only default to True, breaking checkpoint loads)
_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)


# Monkey-patch habitat-sim SimulatorConfiguration for egg ABI compatibility.
# The installed egg has C++ bindings compiled from an older habitat-sim source
# that lacks the `agents` attribute, but the Python wrapper (_sanitize_config)
# expects it. Patch the validator to tolerate missing C++ attributes.
def _patch_habitat_sim() -> None:
    try:
        import habitat_sim.simulator as _sim_mod

        # Check if patch already applied
        if hasattr(_sim_mod, "_agents_patched"):
            return

        _orig_post_init = getattr(_sim_mod.Simulator, "__attrs_post_init__", None)
        if _orig_post_init is None:
            return

        def _safe_post_init(self):
            """Wrapper that catches AttributeError on agents check."""
            try:
                _orig_post_init(self)
            except AttributeError as e:
                if "agents" in str(e):
                    # Old C++ bindings — ignore the agents validation
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "habitat-sim C++ bindings missing 'agents' attribute; "
                        "skipping _sanitize_config agents check."
                    )
                else:
                    raise

        _sim_mod.Simulator.__attrs_post_init__ = _safe_post_init
        _sim_mod._agents_patched = True
    except ImportError:
        pass


_patch_habitat_sim()


DEFAULT_SPLITS = ((0.0, 0.5), (0.5, 1.0))


# Method variant configs for ablation experiments (experiment_plan §5.3, §6.3)
# Each maps a method name to the method_config dict passed to run_episode_two_tier.
METHOD_CONFIGS = {
    # Dev subset variants (Stage 1)
    "D0_backend_only": {
        "use_notebook": False, "use_scene_graph": False,
        "use_active_query": False, "use_rejected_tracking": False,
    },
    "D1_react_loop": {
        "use_notebook": True, "use_scene_graph": False,
        "use_active_query": False, "use_rejected_tracking": False,
    },
    "D2_room_graph": {
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": False, "use_rejected_tracking": False,
    },
    "D3_active_query": {
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": True, "use_rejected_tracking": False,
    },
    "D4_rejected_tracking": {
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": True, "use_rejected_tracking": True,
    },
    # Full method (Stage 2)
    "ours_full": {
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": True, "use_rejected_tracking": True,
    },
    # Ablations (Stage 2 §6.3)
    "A1_wo_notebook": {
        "use_notebook": False, "use_scene_graph": True,
        "use_active_query": True, "use_rejected_tracking": True,
    },
    "A3_wo_room_seg": {
        "use_notebook": True, "use_scene_graph": False,
        "use_active_query": True, "use_rejected_tracking": True,
    },
    "A4_wo_graph": {
        "use_notebook": True, "use_scene_graph": False,
        "use_active_query": True, "use_rejected_tracking": True,
    },
    "A5_wo_active_query": {
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": False, "use_rejected_tracking": True,
    },
    "A6_wo_rejected": {
        "use_notebook": True, "use_scene_graph": True,
        "use_active_query": True, "use_rejected_tracking": False,
    },
}


class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        elapsed_seconds = record.created - self.start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def _setup_logging(output_dir: str, start_ratio: float, end_ratio: float) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging_path = os.path.join(
        output_dir, f"log_{start_ratio:.2f}_{end_ratio:.2f}.log"
    )
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(processName)s - %(message)s")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


def _load_questions(cfg, questions_limit: int) -> list[dict]:
    questions_path = Path(cfg.questions_list_path)
    questions = json.load(open(questions_path, "r", encoding="utf-8"))
    questions = sorted(questions, key=lambda x: x["question_id"])
    if questions_limit > 0:
        questions = questions[:questions_limit]
    return questions


def _select_split(
    questions: list[dict], start_ratio: float, end_ratio: float
) -> list[dict]:
    total_questions = len(questions)
    start_idx = int(start_ratio * total_questions)
    end_idx = int(end_ratio * total_questions)
    return questions[start_idx:end_idx]


def _save_episode_json(
    output_dir: str,
    split_name: str,
    question_id: str,
    payload: dict,
) -> None:
    episode_result_dir = os.path.join(output_dir, "episode_results", split_name)
    os.makedirs(episode_result_dir, exist_ok=True)
    with open(
        os.path.join(episode_result_dir, f"{question_id}.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _save_split_summary(output_dir: str, split_name: str, results: list[dict]) -> None:
    summary_dir = os.path.join(output_dir, "episode_results")
    os.makedirs(summary_dir, exist_ok=True)
    with open(
        os.path.join(summary_dir, f"summary_{split_name}.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


def _load_clip():
    local_clip_path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
    pretrained = local_clip_path if os.path.exists(local_clip_path) else "laion2b_s34b_b79k"
    if os.path.exists(local_clip_path):
        logging.info("Load CLIP weights from local cache: %s", local_clip_path)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return clip_model, clip_preprocess, clip_tokenizer


def main(
    cfg,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    questions_limit: int = 41,
    max_planner_rounds: int = 20,
    aggregate_results: bool = True,
    method_name: str = "ours_full",
    method_config: Optional[dict] = None,
    run_logger: Optional["RunLogger"] = None,
):
    split_name = f"{start_ratio:.2f}_{end_ratio:.2f}"
    _setup_logging(cfg.output_dir, start_ratio, end_ratio)

    logging.info("***** Running Two-Tier AEQA Evaluation *****")
    logging.info("Question path: %s", cfg.questions_list_path)
    logging.info("Output dir: %s", cfg.output_dir)
    logging.info("Split: %.2f-%.2f", start_ratio, end_ratio)
    logging.info("Method: %s config=%s", method_name, method_config or {})

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    questions = _load_questions(cfg, questions_limit)
    total_questions = len(questions)
    split_questions = _select_split(questions, start_ratio, end_ratio)
    logging.info("Total questions loaded for evaluation: %d", total_questions)
    logging.info("Questions in this split: %d", len(split_questions))

    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info("Load YOLO model %s successful!", cfg.yolo_model_name)

    sam_predictor = SAM(cfg.sam_model_name)
    logging.info("Load SAM model %s successful!", cfg.sam_model_name)

    clip_model, clip_preprocess, clip_tokenizer = _load_clip()
    logging.info("Load CLIP model successful!")

    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(split_questions),
        voxel_size=cfg.tsdf_grid_size,
    )

    split_results = []
    for question_idx, question_data in enumerate(split_questions):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]

        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info("Question %s already processed", question_id)
            continue
        if any(invalid_scene_id in scene_id for invalid_scene_id in INVALID_SCENE_ID):
            logging.info("Skip invalid scene %s", scene_id)
            continue

        question = question_data["question"]
        gt_answer = question_data["answer"]
        logging.info("\n========\nIndex: %d Scene: %s", question_idx, scene_id)
        logging.info("Question id %s initialization successful!", question_id)
        logging.info("Question: %s", question)
        logging.info("Ground truth answer: %s", gt_answer)

        episode_dir, _, _, _ = logger.init_episode(
            question_id=question_id,
            init_pts_voxel=(0, 0),
        )

        pts, angle = get_pts_angle_aeqa(
            question_data["position"], question_data["rotation"]
        )
        workflow_output_dir = os.path.join(
            cfg.output_dir, "two_tier_workflow", split_name, question_id
        )

        try:
            result = _run_aeqa_runtime(
                scene_id=scene_id,
                question=question,
                question_id=question_id,
                cfg=cfg,
                detection_model=detection_model,
                sam_predictor=sam_predictor,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
                output_dir=workflow_output_dir,
                max_planner_rounds=max_planner_rounds,
                max_total_steps=cfg.get("two_tier_max_steps", cfg.num_step),
                start_pts=pts,
                start_angle=angle,
                run_logger=run_logger,
                method_config=method_config,
            )
        except Exception as e:
            logging.exception("Episode %s failed with error: %s", question_id, e)
            result = {
                "scene_id": scene_id,
                "question_id": question_id,
                "question": question,
                "answer": "",
                "success": False,
                "steps_taken": 0,
                "rounds_used": 0,
                "error": str(e),
            }

        task_success = bool(result.get("success", False))
        gpt_answer = result.get("answer", "")
        steps_taken = result.get("steps_taken", 0)
        rounds_used = result.get("rounds_used", 0)
        error = result.get("error", "")

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=float(result.get("path_length", 0.0)),
            gpt_answer=gpt_answer,
            n_filtered_snapshots=int(result.get("n_filtered_snapshots", 0)),
            n_total_snapshots=int(result.get("n_total_snapshots", 0)),
            n_total_frames=steps_taken,
        )

        episode_payload = {
        **{k: v for k, v in result.items() if not k.startswith("_")},
            "ground_truth_answer": gt_answer,
            "category": question_data.get("category"),
            "class": question_data.get("class"),
            "split": split_name,
        }
        _save_episode_json(cfg.output_dir, split_name, question_id, episode_payload)
        split_results.append(episode_payload)

        logging.info("Question: %s", question)
        logging.info("Answer: %s", gt_answer)
        logging.info("Prediction: %s", gpt_answer)
        logging.info("Steps: %s, Rounds: %s", steps_taken, rounds_used)
        if error:
            logging.info("Error: %s", error)

        logger.save_results()
        _save_split_summary(cfg.output_dir, split_name, split_results)

        if not cfg.get("save_visualization", False):
            shutil.rmtree(episode_dir, ignore_errors=True)

    logger.save_results()
    if aggregate_results:
        logger.aggregate_results()
    _save_split_summary(cfg.output_dir, split_name, split_results)
    logging.info("Split %.2f-%.2f finished", start_ratio, end_ratio)


def _aggregate_all_results(output_dir: str) -> None:
    output_path = Path(output_dir)

    def split_paths(pattern: str):
        return [
            path for path in sorted(output_path.glob(pattern))
            if not path.stem.endswith("_0.0_1.0")
        ]

    success_list = []
    for path in split_paths("success_list_*.pkl"):
        with open(path, "rb") as fh:
            success_list.extend(pickle.load(fh))
    with open(output_path / "success_list.pkl", "wb") as fh:
        pickle.dump(success_list, fh)
    with open(output_path / "success_list_0.0_1.0.pkl", "wb") as fh:
        pickle.dump(success_list, fh)

    fail_list = []
    for path in split_paths("fail_list_*.pkl"):
        with open(path, "rb") as fh:
            fail_list.extend(pickle.load(fh))
    with open(output_path / "fail_list.pkl", "wb") as fh:
        pickle.dump(fail_list, fh)
    with open(output_path / "fail_list_0.0_1.0.pkl", "wb") as fh:
        pickle.dump(fail_list, fh)

    path_length_list = {}
    for path in split_paths("path_length_list_*.pkl"):
        with open(path, "rb") as fh:
            path_length_list.update(pickle.load(fh))
    with open(output_path / "path_length_list.pkl", "wb") as fh:
        pickle.dump(path_length_list, fh)
    with open(output_path / "path_length_list_0.0_1.0.pkl", "wb") as fh:
        pickle.dump(path_length_list, fh)

    gpt_answer_list = []
    for path in split_paths("gpt_answer_*.json"):
        with open(path, "r", encoding="utf-8") as fh:
            gpt_answer_list.extend(json.load(fh))
    with open(output_path / "gpt_answer.json", "w", encoding="utf-8") as fh:
        json.dump(gpt_answer_list, fh, indent=4, ensure_ascii=False)
    with open(output_path / "gpt_answer_0.0_1.0.json", "w", encoding="utf-8") as fh:
        json.dump(gpt_answer_list, fh, indent=4, ensure_ascii=False)

    for stem in ("n_filtered_snapshots", "n_total_snapshots", "n_total_frames"):
        merged = {}
        for path in split_paths(f"{stem}_*.json"):
            with open(path, "r", encoding="utf-8") as fh:
                merged.update(json.load(fh))
        with open(output_path / f"{stem}.json", "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=4, ensure_ascii=False)
        with open(output_path / f"{stem}_0.0_1.0.json", "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=4, ensure_ascii=False)


def _load_cfg(cfg_file: str, exp_suffix: str, questions_path: Optional[str] = None):
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)
    if questions_path:
        cfg.questions_list_path = questions_path
    cfg.output_dir = os.path.join(
        cfg.output_parent_dir, f"{cfg.exp_name}_{exp_suffix}"
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    shutil.copy2(cfg_file, os.path.join(cfg.output_dir, os.path.basename(cfg_file)))
    return cfg


def _worker_entry(
    cfg_file: str,
    start_ratio: float,
    end_ratio: float,
    questions_limit: int,
    max_planner_rounds: int,
    exp_suffix: str,
    questions_path: Optional[str],
    method_name: str = "ours_full",
    method_config: Optional[dict] = None,
) -> None:
    cfg = _load_cfg(cfg_file, exp_suffix=exp_suffix, questions_path=questions_path)
    # Build a RunLogger per worker (writes to the run's output_dir)
    from src.run_logger import RunLogger
    import datetime
    run_id = f"{method_name}_AEQA_{datetime.datetime.now().strftime('%Y-%m-%d')}_{start_ratio:.2f}_{end_ratio:.2f}"
    rl = RunLogger(
        output_dir=cfg.output_dir,
        run_id=run_id,
        method_name=method_name,
        dataset="AEQA-41",
        model=os.environ.get("MODEL_NAME", "qwen2.5vl-3b-local"),
        seed=cfg.seed,
        config_path=cfg_file,
        use_notebook=method_config.get("use_notebook", True) if method_config else True,
        use_scene_graph=method_config.get("use_scene_graph", True) if method_config else True,
        use_active_query=method_config.get("use_active_query", True) if method_config else True,
        use_rejected_tracking=method_config.get("use_rejected_tracking", True) if method_config else True,
    )
    try:
        main(
            cfg,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            questions_limit=questions_limit,
            max_planner_rounds=max_planner_rounds,
            aggregate_results=False,
            method_name=method_name,
            method_config=method_config,
            run_logger=rl,
        )
    finally:
        rl.close()


def _run_parallel_splits(args) -> int:
    splits = DEFAULT_SPLITS
    method_config = METHOD_CONFIGS.get(args.method, METHOD_CONFIGS["ours_full"])
    processes = []
    for start_ratio, end_ratio in splits:
        process = mp.Process(
            target=_worker_entry,
            name=f"aeqa-{start_ratio:.2f}-{end_ratio:.2f}",
            args=(
                args.cfg_file,
                start_ratio,
                end_ratio,
                args.questions_limit,
                args.max_planner_rounds,
                args.exp_suffix,
                args.questions_path,
                args.method,
                method_config,
            ),
        )
        process.start()
        processes.append(process)

    exit_code = 0
    for process in processes:
        process.join()
        if process.exitcode != 0:
            exit_code = process.exitcode or 1
    if exit_code == 0:
        cfg = _load_cfg(
            args.cfg_file,
            exp_suffix=args.exp_suffix,
            questions_path=args.questions_path,
        )
        _aggregate_all_results(cfg.output_dir)
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf",
        "--cfg_file",
        default="cfg/eval_aeqa.yaml",
        type=str,
        help="AEQA config file path",
    )
    parser.add_argument(
        "--questions_path",
        default=None,
        type=str,
        help="Override questions JSON path. Defaults to cfg.questions_list_path.",
    )
    parser.add_argument(
        "--questions_limit",
        default=41,
        type=int,
        help="Number of sorted AEQA questions to evaluate before ratio splitting.",
    )
    parser.add_argument(
        "--max_planner_rounds",
        default=20,
        type=int,
        help="Maximum two-tier planner rounds per question.",
    )
    parser.add_argument(
        "--start_ratio",
        default=None,
        type=float,
        help="Run a single split starting at this ratio.",
    )
    parser.add_argument(
        "--end_ratio",
        default=None,
        type=float,
        help="Run a single split ending at this ratio.",
    )
    parser.add_argument(
        "--single_split",
        action="store_true",
        help="Run only --start_ratio/--end_ratio instead of the default two parallel splits.",
    )
    parser.add_argument(
        "--exp_suffix",
        default="two_tier_aeqa",
        type=str,
        help="Suffix appended to cfg.exp_name for output_dir.",
    )
    parser.add_argument(
        "--method",
        default="ours_full",
        type=str,
        choices=list(METHOD_CONFIGS.keys()),
        help="Method variant. Controls method_config flags for ablation. "
             "Options: D0_backend_only, D1_react_loop, D2_room_graph, D3_active_query, "
             "D4_rejected_tracking, ours_full, A1_wo_notebook, A3_wo_room_seg, "
             "A4_wo_graph, A5_wo_active_query, A6_wo_rejected.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    method_config = METHOD_CONFIGS.get(args.method, METHOD_CONFIGS["ours_full"])
    single_split = args.single_split or args.start_ratio is not None or args.end_ratio is not None
    if single_split:
        start_ratio = 0.0 if args.start_ratio is None else args.start_ratio
        end_ratio = 1.0 if args.end_ratio is None else args.end_ratio
        cfg = _load_cfg(
            args.cfg_file,
            exp_suffix=args.exp_suffix,
            questions_path=args.questions_path,
        )
        # Build RunLogger for single-split runs
        from src.run_logger import RunLogger
        import datetime
        run_id = f"{args.method}_AEQA_{datetime.datetime.now().strftime('%Y-%m-%d')}_{start_ratio:.2f}_{end_ratio:.2f}"
        rl = RunLogger(
            output_dir=cfg.output_dir,
            run_id=run_id,
            method_name=args.method,
            dataset="AEQA-41",
            model=os.environ.get("MODEL_NAME", "qwen2.5vl-3b-local"),
            seed=cfg.seed,
            config_path=args.cfg_file,
            use_notebook=method_config.get("use_notebook", True),
            use_scene_graph=method_config.get("use_scene_graph", True),
            use_active_query=method_config.get("use_active_query", True),
            use_rejected_tracking=method_config.get("use_rejected_tracking", True),
        )
        try:
            main(
                cfg,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                questions_limit=args.questions_limit,
                max_planner_rounds=args.max_planner_rounds,
                method_name=args.method,
                method_config=method_config,
                run_logger=rl,
            )
        finally:
            rl.close()
    else:
        mp.set_start_method("spawn", force=True)
        raise SystemExit(_run_parallel_splits(args))
