"""8 node functions for the Two-Tier LangGraph state machine.

Each node is a thin wrapper around existing helpers in agent_workflow.py.
The nested closures (`_build_scene_analysis`, `_build_actions`,
`_first_available_action`, `_select_frontier_with_vlm`) from
`run_episode_two_tier` (agent_workflow.py:1255-1435) are lifted to module-level
functions taking explicit args, with logic copied verbatim.

Node signatures use LangGraph's `(state, config) -> partial state dict` convention.
Heavy resources are accessed via `config["configurable"]["resources"]`.

Nodes:
  1. init_node          — episode setup + initial panorama (wraps :1160-1251)
  2. build_context_node — assemble 4-component planner prompt (wraps :1445-1532)
  3. planner_node       — LLM decision + Stage 6.5 frontier sub-selection (:1538-1580)
  3b. critic_node       — (D3) evaluates PlannerAction, can veto + force re-decision
  4. loop_guard_node    — 3 guards + log_decision (wraps :1582-1612)
  5. executor_node      — dispatch action via ToolRegistry (wraps :1633-1635)
  6. memory_update_node — notebook + scene-graph + rejected marking (:1637-1663)
  7. submit_node        — terminal answer, success + fallback paths
"""
from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Optional

import numpy as np

from src.agent_evidence import TrajectoryEvidence
from src.agent_notebook import EvidenceNotebook
from src.agent_planner import PlannerAction, PLANNER_SYSTEM_PROMPT
from src.agent_workflow import (
    RoundTrace,
    _build_current_view_images,
    _build_frontier_mosaic_b64,
    _is_valid_object_desc,
    _parse_stage65_frontier_response,
    _view_idx_from_snapshot_id,
    build_planner_topdown_map_b64,
    call_vlm,
)
from src.scene_graph_memory import SceneGraphMemory

from .edges import after_guard, after_memory  # noqa: F401 — re-exported for tests
from .resources import Resources
from .state import TwoTierState, TransitionReason
from .tools import ToolContext

logger = logging.getLogger(__name__)


# ── Lifted helper functions ──────────────────────────────────────────────
# These were nested closures inside run_episode_two_tier (agent_workflow.py:1255-1435).
# Lifted to module-level taking explicit args. Logic copied verbatim.


def _build_scene_analysis(tsdf_planner, notebook, pts) -> str:
    """Lifted from agent_workflow.py:1255-1277. Logic copied verbatim."""
    lines = ["## Scene Analysis"]
    current_room = (
        tsdf_planner.get_room_id_at(tsdf_planner.habitat2voxel(pts)[:2])
        if pts is not None and hasattr(tsdf_planner, "get_room_id_at")
        else -1
    )
    lines.append(f"Current region: room {current_room}")
    if hasattr(tsdf_planner, "room_regions") and tsdf_planner.room_regions:
        unvisited_rooms = [
            r.room_id for r in tsdf_planner.room_regions
            if r.room_id != current_room and f"seed_{r.room_id}" not in notebook.get_visited_seeds()
        ]
        lines.append(f"Unvisited seed candidates: {', '.join(map(str, unvisited_rooms)) if unvisited_rooms else 'none'}")
    else:
        lines.append("Unvisited seed candidates: none")
    available_frontiers = [
        ft.frontier_id for ft in getattr(tsdf_planner, "frontiers", []) or []
    ][:8]
    lines.append(
        f"Current frontier candidates: {', '.join(map(str, available_frontiers)) if available_frontiers else 'none'}"
    )
    return "\n".join(lines)


def _build_progress(
    round_num: int,
    max_planner_rounds: int,
    question: str,
    notebook: EvidenceNotebook,
    current_views: Optional[list[dict]] = None,
) -> str:
    """Lifted from agent_workflow.py:1279-1297. Logic copied verbatim."""
    lines = [
        "## Progress",
        f"Target: {question}",
        f"Round {round_num + 1} / {max_planner_rounds}",
    ]
    if current_views:
        lines.append("Current-view snapshots attached to this message:")
        for view in current_views:
            lines.append(
                f"- {view['snapshot_id']}: {view['direction']} view, view_idx={view['view_idx']}"
            )
    lines.append(
        "Top-down map: attached to this message; it shows map geometry, frontiers, agent pose, and traversed path only."
    )
    visited = notebook.get_visited_seeds()
    if visited:
        lines.append(f"Already visited: {', '.join(sorted(visited))}")
    return "\n".join(lines)


def _build_reasoning_history(traces: list, max_entries: int = 8) -> str:
    """Lifted from agent_workflow.py:1299-1311. Logic copied verbatim.
    (Original was already a pure function — no closures.)"""
    lines = ["## History", "Reasoning-expectation chain:"]
    if not traces:
        lines.append("- No previous planner rounds.")
    for trace in traces[-max_entries:]:
        lines.extend([
            f"- Round {trace.round_id}",
            f"  action: {trace.action}",
            f"  reason: {trace.reason}",
        ])
        if trace.expected:
            lines.append(f"  expected: {trace.expected}")
    return "\n".join(lines)


def _select_frontier_with_vlm(
    llm_provider,
    tsdf_planner,
    base_action: PlannerAction,
    history_text: str,
    scene_text: str,
    progress_text: str,
) -> PlannerAction:
    """Lifted from agent_workflow.py:1313-1358. Logic copied verbatim.

    Only change: `call_vlm(...)` → `llm_provider.decide_raw(...)` to go through
    the provider abstraction. MimoProvider.decide_raw delegates to call_vlm, so
    behavior is byte-identical.
    """
    frontier_mosaic_b64, frontier_ids = _build_frontier_mosaic_b64(
        getattr(tsdf_planner, "frontiers", [])
    )
    if not frontier_mosaic_b64 or not frontier_ids:
        logger.info("Stage 6.5: no frontier images available; using planner frontier_id=%s", base_action.frontier_id)
        return base_action

    prompt = (
        "Stage 6.5: Frontier Selection\n\n"
        "The Planner chose to explore a frontier. The attached image is a labeled mosaic of all currently alive frontier views.\n"
        f"Current frontier ids: {', '.join(map(str, frontier_ids))}\n\n"
        f"{history_text}\n\n"
        f"{scene_text}\n\n"
        f"{progress_text}\n\n"
        f"Original Planner reason: {base_action.reason}\n"
        f"Original Planner expected: {base_action.expected or 'not specified'}\n\n"
        "Select exactly one frontier_id from the current ids. Respond with valid JSON only:\n"
        '{"reasoning":"why this frontier is most promising","expected":"what exploring it should verify","frontier_id": <id>, "confidence": 0.8}'
    )
    response = llm_provider.decide_raw(
        [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        image_b64=frontier_mosaic_b64,
        max_tokens=1024,
        temperature=0.2,
    )
    try:
        data = _parse_stage65_frontier_response(response, frontier_ids)
        selected_id = int(data["frontier_id"])
        return PlannerAction(
            action_type="explore_frontier",
            frontier_id=str(selected_id),
            reason=data.get("reasoning") or base_action.reason,
            expected=data.get("expected") or base_action.expected,
            confidence=float(data.get("confidence", base_action.confidence or 0.5)),
        )
    except Exception as e:
        logger.warning("Stage 6.5 parse failed: %s; raw=%s", e, response[:300])
        if base_action.frontier_id is None:
            base_action.frontier_id = str(frontier_ids[0])
        return base_action


def _build_actions(tsdf_planner, notebook: EvidenceNotebook, pts) -> str:
    """Lifted from agent_workflow.py:1360-1404. Logic copied verbatim."""
    lines = ["## Actions"]
    lines.append("1. explore_panorama: re-orient with full panorama")
    lines.append(
        '2. navigate_to_object: {"reasoning":"why","expected":"what this should verify","action":"navigate_to_object","arguments":{"snapshot_id":"stepN_viewM","object_name":"visible object"}}'
    )
    visited_seeds = notebook.get_visited_seeds()

    # List available seeds with room info
    if hasattr(tsdf_planner, "room_regions") and tsdf_planner.room_regions:
        current_room = tsdf_planner.get_room_id_at(tsdf_planner.habitat2voxel(pts)[:2]) if pts is not None else -1
        available = [
            r for r in tsdf_planner.room_regions
            if r.room_id != current_room and f"seed_{r.room_id}" not in visited_seeds
        ]
        if available:
            lines.append("Available seeds (explore different rooms):")
            for r in available:
                lines.append(f"  - Seed {r.room_id}")

    # List available frontiers
    if hasattr(tsdf_planner, "frontiers") and tsdf_planner.frontiers:
        available_frontiers = [
            ft for ft in tsdf_planner.frontiers
        ][:8]
        if available_frontiers:
            lines.append("Current frontiers:")
        for ft in available_frontiers:
            lines.append(f"  - Frontier {ft.frontier_id}")

    lines.append("3. explore_frontier <id>: navigate to one of the current frontiers above")
    lines.append(
        '4. submit_answer: {"reasoning":"why","action":"submit_answer","arguments":{"snapshot_id":"stepN_viewM","answer":"final answer"}}'
    )
    lines.append("Rule: if any attached current-view snapshot or history answers the question, choose submit_answer now.")
    lines.append("Rule: navigate_to_object must cite the snapshot_id that shows the target object and include object_name.")
    lines.append("Rule: submit_answer must cite the snapshot_id that supports the answer and include answer.")
    lines.append("Rule: exploration/navigation actions must include reasoning and expected.")
    lines.append("Rule: submit_answer is terminal and should include reasoning, snapshot_id, and answer; expected may be omitted.")
    lines.append("Rule: do not repeat navigate_to_object for an object you already reached; submit or choose a different exploration target.")
    lines.append("Do not select visited seeds or repeat explore_panorama in the same position.")

    if visited_seeds:
        lines.append(f"Unavailable visited seeds: {', '.join(sorted(visited_seeds))}")
    return "\n".join(lines)


def _first_available_action(
    tsdf_planner,
    notebook: EvidenceNotebook,
    pts,
    prefer_non_panorama: bool = False,
) -> PlannerAction:
    """Lifted from agent_workflow.py:1406-1435. Logic copied verbatim."""
    for ft in getattr(tsdf_planner, "frontiers", []) or []:
        return PlannerAction(
            action_type="explore_frontier",
            frontier_id=str(ft.frontier_id),
            reason="Guard fallback: selected first current frontier",
            confidence=0.1,
        )
    visited_seeds = notebook.get_visited_seeds()
    current_room = tsdf_planner.get_room_id_at(tsdf_planner.habitat2voxel(pts)[:2]) if pts is not None else -1
    for room in getattr(tsdf_planner, "room_regions", []) or []:
        if room.room_id != current_room and f"seed_{room.room_id}" not in visited_seeds:
            return PlannerAction(
                action_type="explore_seed",
                seed_id=str(room.room_id),
                reason="Guard fallback: selected first unvisited seed",
                confidence=0.1,
            )
    if prefer_non_panorama:
        return PlannerAction(
            action_type="submit_answer",
            answer="unanswerable",
            reason="Guard fallback: no unvisited navigation target remains",
            confidence=0.0,
        )
    return PlannerAction(
        action_type="explore_panorama",
        reason="Guard fallback: no unvisited navigation target available",
        confidence=0.1,
    )


# ── Node functions ───────────────────────────────────────────────────────


def note_node(state: TwoTierState, config) -> dict:
    """Node 0: 任务分类与拆解。

    AEQA 路径（goal_type=None）：输出 task_type="question"，is_terminal_task=False。
    GOATBench 路径：按 goal_type 分类，结合 cross_subtask_notes 生成计划。

    Phase-1 确定性分类器，不调 LLM。未来 lever：LLM 拆解（Phase C/D）。

    Reads: question, cross_subtask_notes (从 resources 读 goal_type)。
    Writes: task_type, task_plan, is_terminal_task, subtask_index, subtask_total。
    """
    res: Resources = config["configurable"]["resources"]
    goal_type = res.goal_type
    prior_notes = state.get("cross_subtask_notes", [])

    if goal_type is None:
        # AEQA 路径
        return {
            "task_type": "question",
            "task_plan": f"Explore scene, gather evidence, answer: {state['question']}",
            "is_terminal_task": False,
            "subtask_index": 0,
            "subtask_total": 1,
        }

    # GOATBench 路径
    type_map = {
        "object": "object_nav",
        "description": "description_nav",
        "image": "image_nav",
    }
    task_type = type_map.get(goal_type, "object_nav")

    prior_summary = ""
    if prior_notes:
        prior_summary = "\nPrior subtasks found: " + "; ".join(prior_notes[-3:])

    return {
        "task_type": task_type,
        "task_plan": f"Navigate to {goal_type} target: {state['question']}{prior_summary}",
        "is_terminal_task": True,
        "subtask_index": state.get("subtask_index", 0),
        "subtask_total": state.get("subtask_total", 1),
    }


def init_node(state: TwoTierState, config) -> dict:
    """Node 1: episode setup + initial panorama.

    Wraps agent_workflow.py:1160-1251. The heavy resource building (Scene,
    TSDFPlanner, MemoryStore, Executor, Planner, notebook, scene_graph,
    LLMProvider, ToolRegistry) is done by the entrypoint BEFORE graph.invoke;
    this node does the initial panorama (Step 1) and seeds the state.

    Reads from config: resources (fully built).
    Reads from state: scene_id, question_id, question, output_dir, budgets,
                      method flags.
    Writes to state: pose, rounds_used=0, steps_taken, action_history=[],
                     round_traces=[], scene_analysis/history_text/progress_text/
                     actions_text (seeded), current_views, topdown_b64.
    """
    res: Resources = config["configurable"]["resources"]

    # Step 1: Initial panorama (wraps :1228-1251)
    logger.info("--- Two-Tier Step 1: Initial Panorama ---")
    evidence = res.executor.explore_panorama()
    pts, angle = res.executor._pts, res.executor._angle

    from src.agent_tools import silent_perception_step
    current_step = int(getattr(silent_perception_step, "_step_counter", 0))

    res.notebook.update_from_evidence(evidence, step=current_step)

    # Sync scene graph from room segmentation + record initial evidence
    if res.scene_graph is not None:
        res.scene_graph.sync_rooms_from_tsdf(res.tsdf_planner)
        res.scene_graph.add_evidence(
            decision_id=0, action="explore_panorama", outcome=evidence.outcome,
            room_id=evidence.room_id, key_frame_ids=evidence.key_frames,
            objects_nearby=evidence.objects_nearby, progress=evidence.progress,
        )
        res.scene_graph.increment_room_visit(evidence.room_id)

    if res.run_logger is not None:
        res.run_logger.log_trajectory_evidence(
            episode_id=res.question_id, decision_id=0, action="explore_panorama",
            target="initial", outcome=evidence.outcome, room_id=evidence.room_id,
            objects_nearby=evidence.objects_nearby, key_frame_ids=evidence.key_frames,
            steps_taken=current_step,
            progress=evidence.progress,
        )

    # Seed initial prompt artifacts (so state is populated before build_context)
    scene_analysis = _build_scene_analysis(res.tsdf_planner, res.notebook, pts)
    current_views = _build_current_view_images(res.scene, pts, angle, current_step)
    progress = _build_progress(0, state["max_planner_rounds"], state["question"], res.notebook, current_views)
    actions = _build_actions(res.tsdf_planner, res.notebook, pts)
    history = _build_reasoning_history([])

    return {
        "pose": {"pts": pts, "angle": float(angle)},
        "rounds_used": 0,
        "steps_taken": current_step,
        "action_history": [],
        "round_traces": [],
        "scene_analysis": scene_analysis,
        "history_text": history,
        "progress_text": progress,
        "actions_text": actions,
        "current_views": current_views,
        "topdown_b64": None,
        "memory_summary": {},
        "current_action": None,
        "last_evidence": evidence,
        "exhausted_flag": False,
        "answer": "",
        "success": False,
        "error": "",
        "terminal": False,
        "failure_type": "",
    }


def build_context_node(state: TwoTierState, config) -> dict:
    """Node 2: assemble the 4-component planner prompt.

    Wraps agent_workflow.py:1445-1532. Increments rounds_used (mirrors
    `rounds_used = round_num + 1` at :1443). Builds scene_analysis, history,
    progress, actions; runs active memory query; builds topdown map; injects
    structured notebook; writes prompt + topdown to disk.

    Reads: pose, rounds_used, question, max_planner_rounds, method flags,
           round_traces.
    Writes: rounds_used (incremented), scene_analysis, history_text,
            progress_text, actions_text, current_views, topdown_b64,
            memory_summary.
    """
    res: Resources = config["configurable"]["resources"]
    pts = state["pose"]["pts"]
    angle = state["pose"]["angle"]

    # Increment rounds_used (mirrors :1443 `rounds_used = round_num + 1`)
    rounds_used = state["rounds_used"] + 1

    # Build 4-component prompt (wraps :1445-1451)
    history = _build_reasoning_history(state["round_traces"])
    scene_analysis = _build_scene_analysis(res.tsdf_planner, res.notebook, pts)
    current_step_id = max(0, int(getattr(_get_silent_perception_step(), "_step_counter", 0)))
    current_views = _build_current_view_images(res.scene, pts, angle, current_step_id)
    progress = _build_progress(
        rounds_used - 1,  # round_num = rounds_used - 1
        state["max_planner_rounds"],
        state["question"],
        res.notebook,
        current_views,
    )
    actions = _build_actions(res.tsdf_planner, res.notebook, pts)

    # Active memory query (Contribution 3) — wraps :1453-1490
    memory_summary_dict: dict = {}
    if (state["use_active_query"] and res.scene_graph is not None
            and len(res.scene_graph.rooms) > 0):
        query_text = state["question"]
        mq_result = res.scene_graph.query_scene_graph(
            query_text, filters={"status": ["partially_explored", "searched"]},
        )
        if mq_result["candidate_rooms"] or mq_result["candidate_objects"]:
            mem_lines = ["## Memory Query Result"]
            if mq_result["candidate_rooms"]:
                mem_lines.append(f"Candidate rooms from memory: {mq_result['candidate_rooms']}")
            if mq_result["candidate_objects"]:
                obj_cats = []
                for oid in mq_result["candidate_objects"][:5]:
                    obj = res.scene_graph.objects.get(oid)
                    if obj:
                        obj_cats.append(f"{obj.category}(room {obj.room_id}, conf {obj.confidence:.2f})")
                mem_lines.append(f"Remembered objects: {', '.join(obj_cats)}")
            if mq_result["returned_evidence_ids"]:
                mem_lines.append(f"Past evidence: {mq_result['returned_evidence_ids'][:5]}")
            scene_analysis = scene_analysis + "\n" + "\n".join(mem_lines)
            memory_summary_dict = {
                "candidate_rooms": mq_result["candidate_rooms"],
                "candidate_objects": mq_result["candidate_objects"],
                "returned_evidence_ids": mq_result["returned_evidence_ids"],
            }
        if res.run_logger is not None and (mq_result["candidate_rooms"] or mq_result["candidate_objects"]):
            res.run_logger.log_memory_query(
                episode_id=res.question_id, decision_id=rounds_used,
                query_id=mq_result["query_id"], query_text=query_text,
                filters={"status": ["partially_explored", "searched"]},
                candidate_rooms=mq_result["candidate_rooms"],
                candidate_views=mq_result["candidate_views"],
                candidate_objects=mq_result["candidate_objects"],
                returned_evidence_ids=mq_result["returned_evidence_ids"],
                query_latency_sec=mq_result["query_latency_sec"],
            )

    # Write prompt to disk (wraps :1492-1505)
    # C1: L0 visual memory index — always-in-prompt, one line per snapshot.
    l0_text = state.get("l0_index_text", "")
    if l0_text:
        scene_analysis = (
            f"\n[L0 Visual Memory Index]\n{l0_text}\n\n{scene_analysis}"
        )
    # P4: L1 caption layer — CLIP top-K captions injected into scene analysis.
    # TODO: wire real CLIP retrieval. Stub returns all cached captions up to k.
    try:
        from src.two_tier_graph.visual_memory import CaptionStore
        output_dir = state.get("output_dir")
        if output_dir:
            cap_store = CaptionStore(cache_dir=os.path.join(output_dir, "captions"))
            top_captions = cap_store.top_k(query=state["question"], k=3)
            if top_captions:
                cap_block = "\n".join(f"- {c}" for c in top_captions)
                scene_analysis = (
                    f"\n[L1 Visual Memory Captions]\n{cap_block}\n\n{scene_analysis}"
                )
    except Exception:
        logger.warning("Round %d: L1 caption injection failed", rounds_used, exc_info=True)
    prompt_text = res.planner.build_prompt(
        question=state["question"],
        history=history,
        scene=scene_analysis,
        progress=progress,
        actions=actions,
    )
    prompt_dir = os.path.join(state["output_dir"], "planner_prompts")
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_path = os.path.join(prompt_dir, f"round_{rounds_used:02d}_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(prompt_text)
    logger.info(
        "Round %d prompt saved: notebook_entries=%d prompt=%s",
        rounds_used, len(res.notebook.entries), prompt_path,
    )

    # Inject structured notebook into history if enabled (wraps :1507-1509)
    if state["use_notebook"] and res.notebook.structured.hypotheses:
        history = history + "\n" + res.notebook.structured.get_injection_text()

    # Build topdown map (wraps :1511-1532)
    topdown_b64 = build_planner_topdown_map_b64(
        res.memory_store, res.tsdf_planner, pts, angle
    )
    if topdown_b64:
        topdown_dir = os.path.join(state["output_dir"], "planner_topdown")
        os.makedirs(topdown_dir, exist_ok=True)
        with open(
            os.path.join(topdown_dir, f"round_{rounds_used:02d}_topdown.b64.txt"),
            "w", encoding="utf-8",
        ) as fh:
            fh.write(topdown_b64)
        with open(
            os.path.join(topdown_dir, f"round_{rounds_used:02d}_topdown.png"),
            "wb",
        ) as fh:
            fh.write(base64.b64decode(topdown_b64))
    else:
        logger.warning("Round %d: planner topdown map unavailable", rounds_used)

    # ── L2 image recall (P5): token-budgeted original snapshot recall ──
    recalled_views = []
    if state.get("need_visual_recall", False):
        from src.two_tier_graph.visual_memory import ImageRecallStore
        output_dir = state.get("output_dir")
        if output_dir:
            recall_store = ImageRecallStore(cache_dir=os.path.join(output_dir, "recall"))
            loaded_ids = set(state.get("loaded_snapshot_ids", []))
            recalled_views = recall_store.select_for_recall(
                query=state["question"], loaded_ids=loaded_ids
            )
            if recalled_views:
                current_views = current_views + recalled_views
                logger.info(
                    "Round %d: recalled %d snapshots (L2 image recall)",
                    rounds_used, len(recalled_views),
                )

    return {
        "rounds_used": rounds_used,
        "scene_analysis": scene_analysis,
        "history_text": history,
        "progress_text": progress,
        "actions_text": actions,
        "current_views": current_views,
        "topdown_b64": topdown_b64,
        "memory_summary": memory_summary_dict,
    }


def planner_node(state: TwoTierState, config) -> dict:
    """Node 3: LLM decision + Stage 6.5 frontier sub-selection.

    Wraps agent_workflow.py:1538-1580. Calls llm_provider.decide(...) with the
    4-component prompt + images. Fills missing snapshot_id/view_idx. If action
    is explore_frontier, calls _select_frontier_with_vlm (Stage 6.5 sub-call).
    Appends a RoundTrace to round_traces.

    Reads: scene_analysis, history_text, progress_text, actions_text,
           current_views, topdown_b64, question.
    Writes: current_action, round_traces (append).
    """
    res: Resources = config["configurable"]["resources"]
    rounds_used = state["rounds_used"]

    # Build planner images: current views + topdown
    planner_images = [view["image_b64"] for view in state["current_views"]]
    if state["topdown_b64"]:
        planner_images.append(state["topdown_b64"])

    # Planner decides (wraps :1538-1545)
    action = res.llm_provider.decide(
        question=state["question"],
        history=state["history_text"],
        scene=state["scene_analysis"],
        progress=state["progress_text"],
        actions=state["actions_text"],
        image_b64s=planner_images,
    )

    # Snapshot-id fill logic (wraps :1546-1555)
    if action.snapshot_id and action.view_idx is None:
        action.view_idx = _view_idx_from_snapshot_id(action.snapshot_id)
    elif action.action_type in {"navigate_to_object", "submit_answer"} and state["current_views"]:
        action.snapshot_id = state["current_views"][min(1, len(state["current_views"]) - 1)]["snapshot_id"]
        if action.view_idx is None:
            action.view_idx = _view_idx_from_snapshot_id(action.snapshot_id)
        logger.info(
            "Round %d: filled missing snapshot_id with %s for %s",
            rounds_used, action.snapshot_id, action.action_type,
        )

    pts = state["pose"]["pts"]
    logger.info(
        "Round %d: action=%s snapshot=%s confidence=%.2f reason=%s pos=(%.1f,%.1f)",
        rounds_used, action.action_type, action.snapshot_id, action.confidence, action.reason,
        pts[0] if pts is not None else 0.0, pts[2] if pts is not None else 0.0,
    )

    round_trace = RoundTrace(
        round_id=rounds_used,
        action=action.action_type,
        reason=action.reason or "",
        expected=action.expected or "",
    )

    # Stage 6.5 frontier sub-selection (wraps :1567-1580)
    if action.action_type == "explore_frontier":
        action = _select_frontier_with_vlm(
            res.llm_provider,
            res.tsdf_planner,
            action,
            history_text=_build_reasoning_history(state["round_traces"] + [round_trace]),
            scene_text=state["scene_analysis"],
            progress_text=state["progress_text"],
        )
        round_trace.action = action.action_type
        round_trace.reason = action.reason or round_trace.reason
        round_trace.expected = action.expected or round_trace.expected
        logger.info(
            "Round %d Stage 6.5 selected frontier=%s confidence=%.2f reason=%s",
            rounds_used, action.frontier_id, action.confidence, action.reason,
        )

    return {
        "current_action": action,
        "round_traces": [round_trace],  # append via operator.add reducer
    }


def critic_node(state: TwoTierState, config) -> dict:
    """Node 3b (D3): Critic — evaluates PlannerAction, can veto + force re-decision.

    When ``critic.enabled=false`` (default), acts as passthrough (returns empty
    dict → no veto). When enabled, evaluates ``current_action`` and may set
    ``critic_veto=True`` with ``critic_feedback`` to route back to planner for a
    re-decision. Real LLM-based evaluation is a future lever (phase-1 out-of-scope
    #1); stub here never vetoes, so behavior is preserved when flag flips on.

    A/B: critic on vs off, compare accuracy + rounds_used.
    """
    res = config["configurable"]["resources"]

    # Check if critic is enabled (optional attr on Resources; absent → off)
    if not getattr(res, "critic_enabled", False):
        return {}  # passthrough

    action = state.get("current_action")
    if not action:
        return {}

    # TODO: Implement real critic logic (LLM-based evaluation)
    # ponytail: stub never vetoes — swap for LLM critic call when wiring phase-1
    # out-of-scope #1. Ceiling: N veto rounds before forced forward.
    return {"critic_veto": False, "critic_feedback": ""}


def loop_guard_node(state: TwoTierState, config) -> dict:
    """Node 4: apply 3 guards + emit decision trace.

    Wraps agent_workflow.py:1582-1612. Applies three guards (repeated panorama,
    visited seed, invalid object) via _first_available_action. Emits
    run_logger.log_decision with notebook_before (captured here since notebook
    is unchanged between build_context and this point — same as original :1535).

    A NODE (not an edge) because it mutates current_action and must fire the
    decision trace exactly once per round.

    Reads: current_action, action_history, rounds_used, pose, memory_summary.
    Writes: current_action (possibly rewritten).
    """
    res: Resources = config["configurable"]["resources"]
    action: PlannerAction = state["current_action"]
    rounds_used = state["rounds_used"]
    pts = state["pose"]["pts"]

    # Capture notebook_before (mirrors :1535; notebook unchanged since build_context)
    notebook_before = res.notebook.to_dict() if res.run_logger is not None else None

    # ── 3 guards (wrap :1582-1592 verbatim) ──
    if action.action_type == "explore_panorama" and state["action_history"][-1:] == ["explore_panorama"]:
        logger.info("Guard: replacing repeated explore_panorama with unvisited navigation target.")
        action = _first_available_action(res.tsdf_planner, res.notebook, pts, prefer_non_panorama=True)
    elif action.action_type == "explore_seed" and action.seed_id is not None:
        seed_key = f"seed_{action.seed_id}"
        if seed_key in res.notebook.get_visited_seeds():
            logger.info("Guard: replacing visited seed %s.", seed_key)
            action = _first_available_action(res.tsdf_planner, res.notebook, pts, prefer_non_panorama=True)
    elif action.action_type == "navigate_to_object" and not _is_valid_object_desc(action.object_name or ""):
        logger.info("Guard: replacing invalid object navigation target: %s", action.object_name)
        action = _first_available_action(res.tsdf_planner, res.notebook, pts, prefer_non_panorama=True)

    # Record decision trace (Stage 0 logging) — wraps :1594-1611
    if res.run_logger is not None:
        current_room_log = (
            res.tsdf_planner.get_room_id_at(res.tsdf_planner.habitat2voxel(pts)[:2])
            if pts is not None and hasattr(res.tsdf_planner, "get_room_id_at") else -1
        )
        res.run_logger.log_decision(
            episode_id=res.question_id, decision_id=rounds_used,
            current_room=current_room_log,
            notebook_before=notebook_before,
            available_actions=["explore_panorama", "navigate_to_object", "explore_seed", "explore_frontier", "submit_answer"],
            memory_summary=state["memory_summary"],
            planner_reason=action.reason or "",
            selected_action=action.action_type,
            target=action.snapshot_id or action.object_name or action.seed_id or action.frontier_id or "",
            expected_evidence=action.expected or "",
            vlm_calls_this_decision=1,
        )

    return {"current_action": action}


def executor_node(state: TwoTierState, config) -> dict:
    """Node 5: dispatch action via ToolRegistry.

    Wraps agent_workflow.py:1633-1635. Dispatches current_action through the
    ToolRegistry (which delegates to the existing Executor methods 1:1).
    Updates pose from executor._pts/_angle and steps_taken from the global
    step counter.

    Reads: current_action.
    Writes: last_evidence, pose, steps_taken, action_history (append).
    """
    res: Resources = config["configurable"]["resources"]
    action: PlannerAction = state["current_action"]

    ctx = ToolContext(executor=res.executor, resources=res, state=state)
    evidence = res.tool_registry.dispatch(action, ctx)

    from src.agent_tools import silent_perception_step
    steps_taken = int(getattr(silent_perception_step, "_step_counter", 0))

    return {
        "last_evidence": evidence,
        "pose": {"pts": res.executor._pts, "angle": float(res.executor._angle)},
        "steps_taken": steps_taken,
        "action_history": [action.action_type],  # append via operator.add reducer
    }


def memory_update_node(state: TwoTierState, config) -> dict:
    """Node 6: notebook + scene-graph + rejected-region marking.

    Wraps agent_workflow.py:1637-1663. Updates notebook from evidence, syncs
    scene graph, marks rejected regions on GD failure, logs trajectory evidence.
    Computes exhausted_flag for the after_memory edge.

    P0a — Layered compression (Meta-pattern 1):
      L_raw        : round_traces append (done by executor_node); logged here.
      L_compressed : EvidenceNotebook.update_from_evidence, gated by
                     compress_threshold; try/except → 'ok'/'failed'/'skipped'.
      L_index      : stub (C1 will implement); try/except → never blocks.

    Reads: last_evidence, current_action, rounds_used, steps_taken,
           use_scene_graph, use_rejected_tracking, compress_threshold.
    Writes: exhausted_flag, steps_taken (re-synced), compression_log.
    """
    res: Resources = config["configurable"]["resources"]
    evidence: TrajectoryEvidence = state["last_evidence"]
    action: PlannerAction = state["current_action"]
    rounds_used = state["rounds_used"]

    from src.agent_tools import silent_perception_step
    current_step = int(getattr(silent_perception_step, "_step_counter", 0))

    import time
    compression_log = []

    # ── L_raw: round trace already appended by executor_node; record stats ──
    compression_log.append({
        "layer": "L_raw", "round": rounds_used,
        "status": "ok", "input_count": 1, "output_count": 1,
    })

    # ── L_compressed: notebook update, gated by compress_threshold ──
    compress_threshold = state.get("compress_threshold", 5)
    if rounds_used >= compress_threshold:
        try:
            res.notebook.update_from_evidence(evidence, step=current_step)
            compression_log.append({
                "layer": "L_compressed", "round": rounds_used,
                "status": "ok", "input_count": 1, "output_count": 1,
            })
        except Exception as exc:
            logger.exception("L_compressed notebook update failed: %s", exc)
            compression_log.append({
                "layer": "L_compressed", "round": rounds_used,
                "status": "failed", "error": str(exc),
            })
    else:
        compression_log.append({
            "layer": "L_compressed", "round": rounds_used,
            "status": "skipped", "threshold": compress_threshold,
        })

    # Sync scene graph + record trajectory evidence (wraps :1640-1663)
    if res.scene_graph is not None:
        res.scene_graph.sync_rooms_from_tsdf(res.tsdf_planner)
        res.scene_graph.add_evidence(
            decision_id=rounds_used, action=action.action_type,
            outcome=evidence.outcome, room_id=evidence.room_id,
            key_frame_ids=evidence.key_frames,
            objects_nearby=evidence.objects_nearby, progress=evidence.progress,
        )
        if evidence.room_id >= 0:
            res.scene_graph.increment_room_visit(evidence.room_id)
        # Rejected-region tracking (wraps :1651-1653)
        if (state["use_rejected_tracking"]
                and evidence.outcome == "detection_failed"
                and evidence.room_id >= 0):
            res.scene_graph.mark_rejected(evidence.room_id, f"GD failed for {evidence.subgoal}")
            res.notebook.structured.mark_rejected(str(evidence.room_id), f"GD failed for {evidence.subgoal}")

    if res.run_logger is not None:
        res.run_logger.log_trajectory_evidence(
            episode_id=res.question_id, decision_id=rounds_used,
            action=action.action_type,
            target=action.object_name or action.seed_id or action.frontier_id or "",
            outcome=evidence.outcome, room_id=evidence.room_id,
            objects_nearby=evidence.objects_nearby, key_frame_ids=evidence.key_frames,
            success=(evidence.outcome in {"arrived_near_target", "object_found", "panorama_complete"}),
            steps_taken=current_step, progress=evidence.progress,
        )

    # ── L_index: L0 visual memory index (C1) — failure must never block ──
    try:
        from src.two_tier_graph.visual_memory import VisualMemoryIndex, CaptionStore
        refresh_interval = state.get("index_refresh_interval", 3)
        visual_idx = VisualMemoryIndex.from_state(
            state.get("visual_memory_state", {}),
            refresh_interval=refresh_interval,
        )
        # Extract snapshot-equivalent records from this round's evidence.
        # ponytail: TrajectoryEvidence has no pose/object_class fields; we
        # synthesize them from action + state. key_frames are snapshot_ids.
        # When C1b adds a richer snapshot extractor (CLIP ordering), swap here.
        pts = state.get("pose", {}).get("pts", [])
        pose = tuple(pts) if pts is not None else ()
        object_class = action.object_name or evidence.subgoal or "unknown"
        one_line_desc = evidence.progress or evidence.outcome
        snapshot_ids = list(evidence.key_frames) or [f"r{rounds_used}_evidence"]
        loaded = set(state.get("loaded_snapshot_ids", []))
        new_loaded = list(loaded)
        for snap_id in snapshot_ids:
            if snap_id in loaded:
                continue  # cross-round dedup via loaded_snapshot_ids
            visual_idx.update(
                round_idx=rounds_used,
                pose=pose,
                object_class=object_class,
                one_line_desc=one_line_desc,
                snapshot_id=snap_id,
                clip_embedding=None,  # ponytail: C1b wires real CLIP embedding here
            )
            loaded.add(snap_id)
            new_loaded.append(snap_id)
        # ── L1 caption layer (P4): disk-cached VLM caption per snapshot ──
        # TODO: wire real VLM call. Stub uses object_class + one_line_desc.
        output_dir = state.get("output_dir")
        if output_dir:
            cap_store = CaptionStore(cache_dir=os.path.join(output_dir, "captions"))
            for snap_id in snapshot_ids:
                if not cap_store.has(snap_id):
                    # ponytail: placeholder caption — swap for VLM call (e.g.
                    # res.llm_provider.caption(image_b64)) when wired.
                    caption = f"{object_class}: {one_line_desc}"
                    cap_store.put(snap_id, caption)
        if rounds_used % refresh_interval == 0:
            visual_idx._last_rebuild_round = rounds_used
        compression_log.append({
            "layer": "L_index", "round": rounds_used,
            "status": "ok",
            "input_count": len(snapshot_ids),
            "output_count": len(visual_idx._entries),
        })
        updates_l0 = {
            "visual_memory_state": visual_idx.to_state(),
            "l0_index_text": visual_idx.get_index_text(),
            "loaded_snapshot_ids": new_loaded,
        }
    except Exception as exc:
        logger.exception("L_index failed (fallback to L_compressed): %s", exc)
        compression_log.append({
            "layer": "L_index", "round": rounds_used,
            "status": "failed", "error": str(exc),
        })
        updates_l0 = {}

    # Compute exhausted_flag (wraps :1665-1672)
    # NOTE: original uses `action.seed_id or ""` — only seeds, not frontiers.
    exhausted_id = action.seed_id or ""
    exhausted_flag = bool(exhausted_id and res.notebook.is_exhausted(exhausted_id))
    if exhausted_flag:
        logger.info(f"Entity {exhausted_id} exhausted — forcing strategy switch next round.")

    # Per-layer stats to RunLogger (optional, best-effort)
    if res.run_logger is not None and hasattr(res.run_logger, "log_compression_layer"):
        for entry in compression_log:
            res.run_logger.log_compression_layer(
                layer=entry["layer"], round_idx=rounds_used,
                input_count=entry.get("input_count", 0),
                output_count=entry.get("output_count", 0),
                token_est=0, duration=0.0,
            )

    # Determine transition reason (P0b: first-class transition state)
    if rounds_used >= state["max_planner_rounds"]:
        reason = TransitionReason.ROUND_BUDGET
    elif exhausted_flag:
        reason = TransitionReason.EXHAUSTED
    elif current_step >= state["max_total_steps"]:
        reason = TransitionReason.STEP_BUDGET
    else:
        reason = TransitionReason.CONTINUE

    transition = {
        "reason": reason.value,
        "from_node": "memory_update",
        "to_node": "build_context" if reason == TransitionReason.CONTINUE else "submit",
        "round_idx": rounds_used,
    }

    return {
        "exhausted_flag": exhausted_flag,
        "steps_taken": current_step,
        "compression_log": compression_log,
        "last_transition": transition,
        "transition_log": [transition],
        **updates_l0,
    }


def stall_recovery_node(state: TwoTierState, config) -> dict:
    """Node 8 (P3): stall recovery — inject recovery hint into next round.

    Reads ``stall_signal`` (serialized :class:`StallSignal`), converts the hint
    to a ``RecoveryNote`` appended to ``round_traces``, and clears the signal
    so it does not re-trigger on the next ``after_memory`` evaluation.

    Borrows Claude Code's ``transition.reason`` recovery pattern: when the
    planner is stuck repeating the same action with no progress, surface a
    hint rather than silently burning another round.
    """
    signal = state.get("stall_signal")
    if not signal:
        return {}

    hint = signal.get("hint", "Stall detected. Try a different action or object.")
    recovery_note = {
        "round_id": state.get("rounds_used", 0),
        "action": "stall_recovery",
        "reason": hint,
        "expected": "different_action",
    }
    logger.info("Stall recovery: %s", hint)
    return {
        "round_traces": [recovery_note],
        "stall_signal": None,  # clear — do not re-trigger
    }


def submit_node(state: TwoTierState, config) -> dict:
    """Node 7: terminal answer.

    Two entry modes (distinguished by current_action.action_type):
      - Success mode (from after_guard when action is submit_answer):
        wraps :1614-1630. answer = current_action.answer, success = True.
      - Fallback mode (from after_memory when budget exhausted):
        wraps :1681-1702. Calls llm_provider.decide with fallback prompt,
        answer = fallback.answer or "unanswerable",
        success = "unanswerable" not in answer.lower().

    The entry mode is determined by current_action.action_type: if it's
    submit_answer, we're in success mode; otherwise fallback mode. This is safe
    because after_guard routes submit_answer → submit (success), and
    after_memory routes fallback_submit → submit only when current_action is
    NOT submit_answer (otherwise after_guard would have caught it).

    Reads: current_action, question, rounds_used, steps_taken, method flags.
    Writes: answer, success, steps_taken, rounds_used, terminal=True,
            failure_type.
    """
    res: Resources = config["configurable"]["resources"]
    action: PlannerAction = state["current_action"]

    from src.agent_tools import silent_perception_step
    steps_taken = int(getattr(silent_perception_step, "_step_counter", 0))

    if action.action_type == "submit_answer":
        # ── Success path (wraps :1614-1630) ──
        answer = action.answer or ""
        logger.info(f"Answer submitted at round {state['rounds_used']}: {answer}")
        if res.run_logger is not None:
            if action.snapshot_id:
                res.run_logger.register_evidence_id(action.snapshot_id)
            if res.scene_graph is not None:
                res.run_logger.save_graph(res.question_id, res.scene_graph.to_dict())
            res.run_logger.finalize_episode(
                episode_id=res.question_id, success=True,
                answer=answer, evidence_ids=[action.snapshot_id] if action.snapshot_id else None,
                num_steps=int(steps_taken),
            )
        return {
            "answer": answer,
            "success": True,
            "steps_taken": steps_taken,
            "rounds_used": state["rounds_used"],
            "terminal": True,
            "failure_type": "",
        }

    # ── Fallback path (wraps :1681-1702) ──
    # P3: verification nudge — on first fallback entry, give the planner one
    # more round with a verify hint before committing to a best-guess answer.
    # Borrows Claude Code's TodoWrite "verify before done" pattern.
    if not state.get("verification_attempted", False):
        logger.info("Fallback reached — verify nudge: one more round before final submit.")
        return {
            "verification_attempted": True,
            "terminal": False,  # route back to build_context via after_submit edge
            "rounds_used": state["rounds_used"],  # unchanged; nudge doesn't consume a round
        }

    logger.info("Budget exhausted — fallback: submit best guess.")
    fallback_action = res.llm_provider.decide(
        question=state["question"],
        history=res.notebook.get_injection_text(),
        scene="## Scene Analysis\n(No current observation — budget exhausted)",
        progress="## Progress\nAll rounds used. Submit your best answer now.",
        actions="## Actions\n6. submit_answer <your_best_guess>",
        image_b64=None,
    )
    answer = fallback_action.answer or "unanswerable"
    success = "unanswerable" not in answer.lower()

    if res.run_logger is not None:
        failure_type = "premature_submit" if not success else "budget_exhausted_answered"
        if res.scene_graph is not None:
            res.run_logger.save_graph(res.question_id, res.scene_graph.to_dict())
        res.run_logger.finalize_episode(
            episode_id=res.question_id, success=success,
            answer=answer, num_steps=int(steps_taken),
            failure_type=failure_type if not success else "",
            failure_reason="step budget exhausted" if not success else "",
        )

    return {
        "answer": answer,
        "success": success,
        "steps_taken": steps_taken,
        "rounds_used": state["max_planner_rounds"],
        "terminal": True,
        "failure_type": "premature_submit" if not success else "budget_exhausted_answered",
    }


# ── Helper ───────────────────────────────────────────────────────────────


def _get_silent_perception_step():
    """Lazy import to avoid circular deps. Returns the function attribute holder."""
    from src.agent_tools import silent_perception_step
    return silent_perception_step
