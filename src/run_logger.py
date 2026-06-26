"""Stage 0 RunLogger — structured experiment trace recording.

Implements the logging schema defined in experiment_plan_AEQA_to_GOATBench.md §4.
Every run produces:
  results/<method>_<dataset>_<date>/
    run_manifest.json         — run metadata (method, model, code version, seed)
    episode_metrics.csv       — per-episode metrics (success, steps, vlm calls, ...)
    decision_trace.jsonl      — one record per planner decision
    memory_query_trace.jsonl  — one record per active memory query
    room_view_object_graph.json — final scene graph dump
    trajectory_evidence.jsonl — one record per executor outcome
    answer_evidence.json      — final answer + supporting evidence ids
    failures.csv              — failures with failure_type classification

Usage:
    run_logger = RunLogger(output_dir, run_id="ours_full_AEQA-41_2026-06-26",
                           method_name="ours_full", dataset="AEQA-41", ...)
    run_logger.log_decision(episode_id=..., decision_id=..., ...)
    run_logger.log_memory_query(episode_id=..., decision_id=..., ...)
    run_logger.log_trajectory_evidence(episode_id=..., ...)
    run_logger.log_episode_metrics(episode_id=..., success=..., llm_match=..., ...)
    run_logger.finalize_episode(episode_id=..., answer=..., evidence_ids=...)
    run_logger.save_graph(episode_id=..., graph_dict=...)
    run_logger.close()
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunManifest:
    """run_manifest.json schema (§4.3)."""
    run_id: str
    method_name: str
    dataset: str
    model: str
    prompt_version: str
    code_version: str
    seed: int
    config_path: str
    start_time: str
    end_time: str = ""
    # Method component flags (for ablation tracking)
    use_notebook: bool = True
    use_scene_graph: bool = True
    use_active_query: bool = True
    use_rejected_tracking: bool = True
    choose_every_step: bool = False
    # Aggregate counters
    total_episodes: int = 0
    total_decisions: int = 0
    total_memory_queries: int = 0
    total_vlm_calls: int = 0


class RunLogger:
    """Structured trace logger for one experiment run.

    One RunLogger corresponds to one results/<run_id>/ directory. It is
    episode-aware: call start_episode / finalize_episode around each episode,
    and log_decision / log_memory_query / log_trajectory_evidence in between.
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        method_name: str,
        dataset: str,
        model: str,
        prompt_version: str = "v1",
        code_version: str = "",
        seed: int = 77,
        config_path: str = "",
        use_notebook: bool = True,
        use_scene_graph: bool = True,
        use_active_query: bool = True,
        use_rejected_tracking: bool = True,
        choose_every_step: bool = False,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.manifest = RunManifest(
            run_id=run_id,
            method_name=method_name,
            dataset=dataset,
            model=model,
            prompt_version=prompt_version,
            code_version=code_version or self._git_commit(),
            seed=seed,
            config_path=config_path,
            start_time=datetime.now().isoformat(timespec="seconds"),
            use_notebook=use_notebook,
            use_scene_graph=use_scene_graph,
            use_active_query=use_active_query,
            use_rejected_tracking=use_rejected_tracking,
            choose_every_step=choose_every_step,
        )
        # Persist manifest immediately (start_time recorded, end_time updated on close)
        self._save_manifest()

        # Open file handles for streaming jsonl/csv writes
        self._decision_fh = open(
            os.path.join(output_dir, "decision_trace.jsonl"), "a", encoding="utf-8"
        )
        self._memory_query_fh = open(
            os.path.join(output_dir, "memory_query_trace.jsonl"), "a", encoding="utf-8"
        )
        self._trajectory_fh = open(
            os.path.join(output_dir, "trajectory_evidence.jsonl"), "a", encoding="utf-8"
        )
        self._failures_fh = open(
            os.path.join(output_dir, "failures.csv"), "a", encoding="utf-8", newline=""
        )
        self._failures_writer = csv.writer(self._failures_fh)
        if self._failures_fh.tell() == 0:
            self._failures_writer.writerow(
                ["episode_id", "question_or_goal", "failure_type", "reason", "rounds_used", "steps_taken"]
            )

        # episode_metrics is buffered in memory and written at close() to avoid
        # partial-row issues; also written incrementally per-episode for safety.
        self._metrics_path = os.path.join(output_dir, "episode_metrics.csv")
        self._metrics_fields = [
            "episode_id", "question_or_goal", "success", "llm_match", "llm_match_spl",
            "sr", "spl", "path_length", "num_steps", "num_decisions",
            "num_vlm_calls", "num_memory_queries", "num_evidence_viewed",
            "num_rooms_visited", "num_revisited_rooms",
            "final_evidence_ids", "failure_type",
        ]
        self._metrics_rows: list[dict] = []
        # Write header if new
        if not os.path.exists(self._metrics_path) or os.path.getsize(self._metrics_path) == 0:
            with open(self._metrics_path, "w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._metrics_fields).writeheader()

        # Per-episode accumulators (reset in start_episode)
        self._current_episode: Optional[str] = None
        self._episode_vlm_calls = 0
        self._episode_memory_queries = 0
        self._episode_decisions = 0
        self._episode_evidence_viewed = 0
        self._episode_rooms_visited: list[int] = []
        self._episode_evidence_ids: list[str] = []
        self._episode_start_time = 0.0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode_id: str, question_or_goal: str = "") -> None:
        """Begin recording a new episode. Resets per-episode counters."""
        self._current_episode = episode_id
        self._episode_vlm_calls = 0
        self._episode_memory_queries = 0
        self._episode_decisions = 0
        self._episode_evidence_viewed = 0
        self._episode_rooms_visited = []
        self._episode_evidence_ids = []
        self._episode_start_time = time.time()
        self._episode_question = question_or_goal
        logger.info("RunLogger start episode %s", episode_id)

    def finalize_episode(
        self,
        episode_id: str,
        success: bool,
        answer: str = "",
        evidence_ids: Optional[list[str]] = None,
        failure_type: str = "",
        failure_reason: str = "",
        path_length: float = 0.0,
        num_steps: int = 0,
        llm_match: Optional[float] = None,
        llm_match_spl: Optional[float] = None,
        sr: Optional[float] = None,
        spl: Optional[float] = None,
    ) -> None:
        """Finalize an episode: write metrics row, record failure if any, save evidence."""
        if evidence_ids is None:
            evidence_ids = list(self._episode_evidence_ids)
        num_rooms = len(set(self._episode_rooms_visited))
        num_revisited = len(self._episode_rooms_visited) - num_rooms
        row = {
            "episode_id": episode_id,
            "question_or_goal": getattr(self, "_episode_question", ""),
            "success": int(success),
            "llm_match": llm_match if llm_match is not None else "",
            "llm_match_spl": llm_match_spl if llm_match_spl is not None else "",
            "sr": sr if sr is not None else "",
            "spl": spl if spl is not None else "",
            "path_length": f"{path_length:.3f}",
            "num_steps": num_steps,
            "num_decisions": self._episode_decisions,
            "num_vlm_calls": self._episode_vlm_calls,
            "num_memory_queries": self._episode_memory_queries,
            "num_evidence_viewed": self._episode_evidence_viewed,
            "num_rooms_visited": num_rooms,
            "num_revisited_rooms": max(0, num_revisited),
            "final_evidence_ids": ";".join(evidence_ids),
            "failure_type": failure_type,
        }
        self._metrics_rows.append(row)
        # Incremental write (append)
        with open(self._metrics_path, "a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._metrics_fields).writerow(row)

        # answer_evidence.json (one per episode, keyed by episode_id)
        ans_path = os.path.join(self.output_dir, "answer_evidence.json")
        ans_data: dict = {}
        if os.path.exists(ans_path):
            try:
                with open(ans_path, "r", encoding="utf-8") as fh:
                    ans_data = json.load(fh)
            except json.JSONDecodeError:
                ans_data = {}
        ans_data[episode_id] = {
            "answer": answer,
            "success": success,
            "evidence_ids": evidence_ids,
            "num_vlm_calls": self._episode_vlm_calls,
            "num_memory_queries": self._episode_memory_queries,
            "num_decisions": self._episode_decisions,
            "latency_sec": round(time.time() - self._episode_start_time, 2),
        }
        with open(ans_path, "w", encoding="utf-8") as fh:
            json.dump(ans_data, fh, indent=2, ensure_ascii=False)

        if not success and failure_type:
            self._failures_writer.writerow([
                episode_id, getattr(self, "_episode_question", ""),
                failure_type, failure_reason,
                self._episode_decisions, num_steps,
            ])
            self._failures_fh.flush()

        self.manifest.total_episodes += 1
        self.manifest.total_decisions += self._episode_decisions
        self.manifest.total_memory_queries += self._episode_memory_queries
        self.manifest.total_vlm_calls += self._episode_vlm_calls
        self._current_episode = None

    # ------------------------------------------------------------------
    # Per-decision / per-query / per-evidence logging
    # ------------------------------------------------------------------

    def log_decision(
        self,
        episode_id: str,
        decision_id: int,
        current_room: Any = None,
        notebook_before: Optional[dict] = None,
        available_actions: Optional[list] = None,
        memory_summary: Optional[dict] = None,
        planner_reason: str = "",
        selected_action: str = "",
        target: str = "",
        expected_evidence: str = "",
        notebook_update: Optional[dict] = None,
        vlm_calls_this_decision: int = 1,
    ) -> None:
        """Record one planner decision (§4.3 decision_trace.jsonl)."""
        record = {
            "episode_id": episode_id,
            "decision_id": decision_id,
            "current_room": current_room,
            "notebook_before": notebook_before or {},
            "available_actions": available_actions or [],
            "memory_summary": memory_summary or {},
            "planner_reason": planner_reason,
            "selected_action": selected_action,
            "target": target,
            "expected_evidence": expected_evidence,
            "notebook_update": notebook_update or {},
        }
        self._decision_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._decision_fh.flush()
        self._episode_decisions += 1
        self._episode_vlm_calls += vlm_calls_this_decision
        self.manifest.total_vlm_calls += vlm_calls_this_decision

    def log_memory_query(
        self,
        episode_id: str,
        decision_id: int,
        query_id: int,
        query_text: str,
        filters: Optional[dict] = None,
        candidate_rooms: Optional[list] = None,
        candidate_views: Optional[list] = None,
        candidate_objects: Optional[list] = None,
        returned_evidence_ids: Optional[list] = None,
        evidence_viewed_by_planner: Optional[list] = None,
        query_latency_sec: float = 0.0,
    ) -> None:
        """Record one active memory query (§4.3 memory_query_trace.jsonl)."""
        record = {
            "episode_id": episode_id,
            "decision_id": decision_id,
            "query_id": query_id,
            "query_text": query_text,
            "filters": filters or {},
            "candidate_rooms": candidate_rooms or [],
            "candidate_views": candidate_views or [],
            "candidate_objects": candidate_objects or [],
            "returned_evidence_ids": returned_evidence_ids or [],
            "evidence_viewed_by_planner": evidence_viewed_by_planner or [],
            "query_latency_sec": round(query_latency_sec, 3),
        }
        self._memory_query_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._memory_query_fh.flush()
        self._episode_memory_queries += 1
        if evidence_viewed_by_planner:
            self._episode_evidence_viewed += len(evidence_viewed_by_planner)

    def log_trajectory_evidence(
        self,
        episode_id: str,
        decision_id: int,
        action: str,
        target: str,
        outcome: str,
        room_id: Any = None,
        objects_nearby: Optional[list] = None,
        key_frame_ids: Optional[list] = None,
        success: bool = False,
        steps_taken: int = 0,
        progress: str = "",
    ) -> None:
        """Record one executor trajectory evidence (§4.3 trajectory_evidence.jsonl)."""
        record = {
            "episode_id": episode_id,
            "decision_id": decision_id,
            "action": action,
            "target": target,
            "outcome": outcome,
            "room_id": room_id,
            "objects_nearby": objects_nearby or [],
            "key_frame_ids": key_frame_ids or [],
            "success": success,
            "steps_taken": steps_taken,
            "progress": progress,
        }
        self._trajectory_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trajectory_fh.flush()
        if room_id is not None and room_id >= 0:
            self._episode_rooms_visited.append(int(room_id))
        if key_frame_ids:
            self._episode_evidence_ids.extend(key_frame_ids)

    def register_evidence_id(self, evidence_id: str) -> None:
        """Manually register an evidence id (e.g. snapshot_id cited in submit_answer)."""
        self._episode_evidence_ids.append(evidence_id)

    def register_vlm_call(self, n: int = 1) -> None:
        """Manually account VLM calls not tied to a decision (e.g. frontier selection)."""
        self._episode_vlm_calls += n
        self.manifest.total_vlm_calls += n

    def save_graph(self, episode_id: str, graph: dict) -> None:
        """Save the room-view-object scene graph for one episode.

        Multiple episodes are merged into a dict keyed by episode_id.
        """
        path = os.path.join(self.output_dir, "room_view_object_graph.json")
        data: dict = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except json.JSONDecodeError:
                data = {}
        data[episode_id] = graph
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalize the run: update manifest end_time, flush and close handles."""
        self.manifest.end_time = datetime.now().isoformat(timespec="seconds")
        self._save_manifest()
        for fh in (
            self._decision_fh, self._memory_query_fh, self._trajectory_fh,
            self._failures_fh,
        ):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_manifest(self) -> None:
        path = os.path.join(self.output_dir, "run_manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self.manifest), fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _git_commit() -> str:
        try:
            import subprocess
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    # Context-manager support for clean teardown
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
