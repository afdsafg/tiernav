"""run_episode_two_tier_langgraph — LangGraph entrypoint.

Exact same signature and return dict as `run_episode_two_tier`
(agent_workflow.py:1087-1104). Builds Resources (heavy perception stack +
memory stores + planner/executor + LLMProvider + ToolRegistry), constructs
TwoTierState, invokes the compiled graph, and maps the terminal state to the
result dict.

The heavy init (Scene, TSDFPlanner, MemoryStore, Executor, Planner, notebook,
scene_graph) mirrors `:1160-1226` verbatim. If init fails, returns the error
result without invoking the graph (mirrors `:1223-1226`). Workflow errors are
caught and mapped to the error result (mirrors `:1705-1722`). scene.__del__()
runs in finally (mirrors `:1724-1729`).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np

from src.agent_executor import Executor
from src.agent_memory import MemoryStore
from src.agent_notebook import EvidenceNotebook
from src.agent_planner import Planner
from src.scene_graph_memory import SceneGraphMemory

from .graph import build_two_tier_graph
from .providers import build_llm_provider
from .resources import Resources
from .state import TwoTierState
from .tools import build_default_tool_registry

logger = logging.getLogger(__name__)


def run_episode_two_tier_langgraph(
    scene_id: str,
    question: str,
    question_id: str,
    cfg,
    detection_model,
    sam_predictor,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    output_dir: str = "/root/MyAgent/results/hmge",
    max_planner_rounds: int = 10,
    max_total_steps: int = 50,
    start_pts: Optional[np.ndarray] = None,
    start_angle: float = 0.0,
    run_logger=None,
    method_config: Optional[dict] = None,
    # ── GOATBench 新增（AEQA 路径不传，走默认值）──
    scene=None,
    tsdf_planner=None,
    notebook=None,
    scene_graph=None,
    goal_type: Optional[str] = None,
    goal_metadata: Optional[dict] = None,
    subtask_index: int = 0,
    subtask_total: int = 1,
    cross_subtask_notes: Optional[list] = None,
) -> Dict:
    """LangGraph-based Two-Tier Planner-Executor episode loop.

    Signature and return dict are identical to `run_episode_two_tier`
    (agent_workflow.py:1087-1104), so this is a drop-in replacement selectable
    via `--engine langgraph`.

    Returns: dict with keys:
        scene_id, question_id, question, answer, success, steps_taken,
        rounds_used, error
    """
    # Method config (ablation flags) — mirrors :1128-1132
    mc = method_config or {}
    use_notebook = mc.get("use_notebook", True)
    use_scene_graph = mc.get("use_scene_graph", True)
    use_active_query = mc.get("use_active_query", True)
    use_rejected_tracking = mc.get("use_rejected_tracking", True)

    # P0a: layered compression config (defaults preserve prior behavior)
    compress_threshold = mc.get("compress_threshold", 5)
    index_refresh_interval = mc.get("index_refresh_interval", 3)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== Two-Tier (LangGraph) Episode {question_id}: {scene_id} ===")
    logger.info(f"Question: {question}")

    result: Dict = {
        "scene_id": scene_id,
        "question_id": question_id,
        "question": question,
        "answer": "",
        "success": False,
        "steps_taken": 0,
        "rounds_used": 0,
        "error": "",
    }

    # ── Build Resources (mirrors :1160-1226) ──────────────────────────
    # GOATBench: caller may inject scene/tsdf_planner/notebook/scene_graph;
    # ownership of scene lifecycle stays with caller in that case.
    scene_owned = scene is None
    try:
        from src.agent_tools import silent_perception_step
        silent_perception_step._last_pos = None
        silent_perception_step._step_counter = -1

        from omegaconf import OmegaConf

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif hasattr(cfg, "concept_graph_config_path"):
            pass
        else:
            from easydict import EasyDict
            cfg = EasyDict(cfg)

        graph_cfg_path = getattr(cfg, "concept_graph_config_path", None)
        if graph_cfg_path and os.path.exists(graph_cfg_path):
            graph_cfg = OmegaConf.load(graph_cfg_path)
            OmegaConf.resolve(graph_cfg)
        else:
            graph_cfg = getattr(cfg, "scene_graph", {})

        if scene is None:
            from src.scene_aeqa import Scene
            scene = Scene(
                scene_id=scene_id, cfg=cfg, graph_cfg=graph_cfg,
                detection_model=detection_model, sam_predictor=sam_predictor,
                clip_model=clip_model, clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
            )

        if start_pts is not None and not np.isnan(start_pts).any():
            pts = start_pts.copy()
            angle = start_angle
        else:
            start_pts_random = scene.pathfinder.get_random_navigable_point()
            if np.isnan(start_pts_random).any():
                start_pts_random = np.array([0.0, 1.5, 0.0])
            pts = start_pts_random.copy()
            angle = 0.0

        if tsdf_planner is None:
            from src.geom import get_scene_bnds
            from src.tsdf_planner import TSDFPlanner
            vol_bnds, _ = get_scene_bnds(scene.pathfinder, floor_height=pts[1])
            tsdf_planner = TSDFPlanner(
                vol_bnds=vol_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=pts[1],
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=bool(getattr(cfg, "save_visualization", False)),
            )

        memory_store = MemoryStore(
            output_dir=os.path.join(output_dir, f"memory_{question_id}")
        )

        executor = Executor(
            scene, tsdf_planner, memory_store, cfg,
            detection_model, sam_predictor,
            clip_model, clip_preprocess, clip_tokenizer,
        )
        executor.set_state(pts, angle, 0)

        from src.const import QWEN_PLANNER_API_KEY, QWEN_PLANNER_BASE_URL
        planner = Planner(api_key=QWEN_PLANNER_API_KEY, base_url=QWEN_PLANNER_BASE_URL, goal_type=goal_type)

        if notebook is None:
            notebook = EvidenceNotebook()
        if scene_graph is None and use_scene_graph:
            scene_graph = SceneGraphMemory()
        elif scene_graph is None and not use_scene_graph:
            scene_graph = None

        # Build LLM provider + tool registry (new abstractions)
        llm_provider = build_llm_provider(cfg, planner)
        tool_registry = build_default_tool_registry()

        resources = Resources(
            scene=scene,
            tsdf_planner=tsdf_planner,
            memory_store=memory_store,
            models={
                "detection": detection_model,
                "sam": sam_predictor,
                "clip": clip_model,
                "clip_preprocess": clip_preprocess,
                "clip_tokenizer": clip_tokenizer,
            },
            cfg=cfg,
            notebook=notebook,
            scene_graph=scene_graph,
            planner=planner,
            executor=executor,
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            run_logger=run_logger,
            question_id=question_id,
            question=question,
            output_dir=output_dir,
            goal_type=goal_type,
            goal_metadata=goal_metadata,
        )

    except Exception as e:
        logger.error(f"Two-tier (LangGraph) initialization failed: {e}")
        result["error"] = str(e)
        if scene is not None and scene_owned:
            try:
                scene.__del__()
            except Exception:
                pass
        return result

    # ── Invoke graph (mirrors :1228-1703 try/except) ──────────────────
    if run_logger is not None:
        run_logger.start_episode(episode_id=question_id, question_or_goal=question)

    try:
        initial_state: TwoTierState = {
            # Episode identity
            "scene_id": scene_id,
            "question_id": question_id,
            "question": question,
            "output_dir": output_dir,
            # Budgets
            "max_planner_rounds": max_planner_rounds,
            "max_total_steps": max_total_steps,
            # Method flags
            "use_notebook": use_notebook,
            "use_scene_graph": use_scene_graph,
            "use_active_query": use_active_query,
            "use_rejected_tracking": use_rejected_tracking,
            # P0a: layered compression
            "compress_threshold": compress_threshold,
            "index_refresh_interval": index_refresh_interval,
            "l0_index_text": "",
            "compression_log": [],
            # Mutable per-round (will be set by init_node)
            "pose": {"pts": pts, "angle": float(angle)},
            "rounds_used": 0,
            "steps_taken": 0,
            "current_action": None,
            "last_evidence": None,
            "exhausted_flag": False,
            # Accumulating history
            "action_history": [],
            "round_traces": [],
            # Per-round prompt artifacts
            "scene_analysis": "",
            "history_text": "",
            "progress_text": "",
            "actions_text": "",
            "current_views": [],
            "topdown_b64": None,
            "memory_summary": {},
            # Terminal
            "answer": "",
            "success": False,
            "error": "",
            "terminal": False,
            "failure_type": "",
            # ── GOATBench 任务上下文（note_node 设置 task_type/task_plan）──
            "task_type": "",            # note_node 会设置
            "task_plan": "",
            "is_terminal_task": False,
            "subtask_index": subtask_index,
            "subtask_total": subtask_total,
            "cross_subtask_notes": cross_subtask_notes or [],
            "observed_goal_positions": [],
            "within_target": False,
            "agent_target_distance": float("inf"),
        }

        graph = build_two_tier_graph()
        # LangGraph default recursion_limit is 25, which is too low for VLN
        # episodes. Each round visits ~5 nodes (build_context -> planner ->
        # loop_guard -> executor -> memory_update), plus init + submit.
        # Set limit = max_planner_rounds * 6 + 20 (generous buffer).
        recursion_limit = max_planner_rounds * 6 + 20
        final_state = graph.invoke(
            initial_state,
            config={
                "configurable": {"resources": resources},
                "recursion_limit": recursion_limit,
            },
        )

        # Map terminal state to result dict
        result["answer"] = final_state.get("answer", "")
        result["success"] = bool(final_state.get("success", False))
        result["steps_taken"] = int(final_state.get("steps_taken", 0))
        result["rounds_used"] = int(final_state.get("rounds_used", 0))
        # Path length: the Executor accumulates distance during navigation
        result["path_length"] = float(getattr(resources.executor, "path_length", 0.0))
        # Snapshot counts: not yet implemented in LangGraph pipeline.
        # TODO: wire snapshot tracking from executor/memory-update nodes.
        result["n_filtered_snapshots"] = 0
        result["n_total_snapshots"] = 0
        # ── Threaded resources + GOATBench return fields ──
        # _-prefixed: non-serializable heavy objects (filtered out of JSON by runners)
        result["_notebook"] = resources.notebook
        result["_scene_graph"] = resources.scene_graph
        # final_pts: ndarray → list for JSON safety; GOATBench runner converts back
        _pts = final_state.get("pose", {}).get("pts")
        result["final_pts"] = _pts.tolist() if hasattr(_pts, "tolist") else _pts
        result["final_angle"] = float(final_state.get("pose", {}).get("angle", 0.0))
        result["cross_subtask_notes"] = final_state.get(
            "cross_subtask_notes", cross_subtask_notes or []
        )
        return result

    except Exception as e:
        logger.error(f"Two-tier (LangGraph) workflow error: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        if run_logger is not None:
            if scene_graph is not None:
                try:
                    run_logger.save_graph(question_id, scene_graph.to_dict())
                except Exception:
                    pass
            run_logger.finalize_episode(
                episode_id=question_id, success=False,
                answer=result.get("answer", ""), num_steps=0,
                failure_type="planner_error",
                failure_reason=str(e)[:200],
            )
        return result

    finally:
        if scene is not None and scene_owned:
            try:
                scene.__del__()
            except Exception:
                pass
