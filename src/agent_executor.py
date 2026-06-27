"""Executor: wraps 6 structured tools over the existing agent_tools layer.

Each tool returns a TrajectoryEvidence instance that compresses the low-level
outcome for the upper-tier Planner / EvidenceNotebook.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.agent_evidence import TrajectoryEvidence

logger = logging.getLogger(__name__)


class Executor:
    """Dispatches PlannerAction to the appropriate low-level tool."""

    def __init__(
        self,
        scene,
        tsdf_planner,
        memory_store,
        cfg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    ):
        self.scene = scene
        self.tsdf = tsdf_planner
        self.memory = memory_store
        self.cfg = cfg
        self.models = {
            "detection": detection_model,
            "sam": sam_predictor,
            "clip": clip_model,
            "clip_preprocess": clip_preprocess,
            "clip_tokenizer": clip_tokenizer,
        }
        self._pts = None
        self._angle = None
        self._step_counter = 0
        self._path_length = 0.0
        self._last_panorama_views = []

    # ── state ─────────────────────────────────────────────────────────

    def set_state(self, pts, angle, step_counter: int):
        self._pts = pts
        self._angle = angle
        self._step_counter = step_counter

    @property
    def path_length(self) -> float:
        return self._path_length

    # ── helpers ───────────────────────────────────────────────────────

    def _m(self) -> dict:
        return self.models

    def _sync_step_counter(self):
        try:
            from src.agent_tools import silent_perception_step
            self._step_counter = int(getattr(silent_perception_step, "_step_counter", self._step_counter))
        except Exception:
            pass

    def _collect_nearby(self, pts) -> list:
        if not hasattr(self.scene, "objects"):
            return []
        nearby = []
        for obj in self.scene.objects.values():
            if not isinstance(obj, dict) or "class_name" not in obj:
                continue
            name = obj["class_name"]
            # Try to get position from bbox or point_cloud
            obj_pos = None
            if "bbox" in obj and obj["bbox"] is not None and hasattr(obj["bbox"], "center"):
                obj_pos = obj["bbox"].center
            elif "point_cloud" in obj:
                obj_pos = obj["point_cloud"].mean(axis=0)
            if obj_pos is not None:
                import numpy as np
                dist = np.linalg.norm(obj_pos[[0, 2]] - pts[[0, 2]])
                if dist < 3.0:  # within 3m
                    nearby.append(name)
        return list(set(nearby))

    # ── 6 tools ───────────────────────────────────────────────────────

    def explore_panorama(self, config: Optional[dict] = None) -> TrajectoryEvidence:
        from src.agent_tools import observe_panorama

        old_pts = self._pts
        pts, angle, _mosaic_b64, text, panorama_views = observe_panorama(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            self._step_counter,
            self.memory,
            self.scene.cam_intrinsic,
            self.cfg,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
        )
        self._pts, self._angle = pts, angle
        if self._pts is not None and old_pts is not None:
            self._path_length += float(np.linalg.norm(np.asarray(self._pts) - np.asarray(old_pts)))
        self._last_panorama_views = panorama_views or []
        self._sync_step_counter()

        room_id = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )
        return TrajectoryEvidence(
            subgoal="Explore panorama for re-orientation",
            task_mode="explore_panorama",
            progress=text,
            salient=[text],
            outcome="panorama_complete",
            current_image_b64=_mosaic_b64,
            room_id=room_id,
            objects_nearby=self._collect_nearby(pts),
        )

    def navigate_to_object(
        self, object_name: str, view_idx: Optional[int] = None
    ) -> TrajectoryEvidence:
        from src.agent_tools import navigate_to_object

        selected_view = None
        if view_idx is not None:
            try:
                idx = int(view_idx)
                selected_view = next(
                    (v for v in self._last_panorama_views if int(v.get("view_idx", -1)) == idx),
                    None,
                )
            except (TypeError, ValueError):
                selected_view = None
        view_angle = selected_view.get("angle") if selected_view else self._angle
        view_cam_pose = selected_view.get("cam_pose") if selected_view else None

        old_pts = self._pts
        pts, angle, success, status, _img = navigate_to_object(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            view_idx,
            view_angle,
            view_cam_pose,
            object_name,
            self.memory,
            self.scene.cam_intrinsic,
            self.cfg,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
            self._step_counter,
        )
        self._pts, self._angle = pts, angle
        if self._pts is not None and old_pts is not None:
            self._path_length += float(np.linalg.norm(np.asarray(self._pts) - np.asarray(old_pts)))
        self._sync_step_counter()

        gd_quality = (
            "ok"
            if success
            else ("detection_failed" if "GD" in status else "target_not_reached")
        )
        room_id = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )

        return TrajectoryEvidence(
            subgoal=f"Navigate to {object_name} via view {view_idx}",
            task_mode="navigate_to_object",
            progress=f"Navigation status: {status}; moved={self._movement(old_pts, pts):.2f}m",
            salient=[object_name, status],
            outcome="arrived_near_target" if success else "target_not_reached",
            gd_quality=gd_quality,
            current_image_b64=_img,
            room_id=room_id,
            objects_nearby=self._collect_nearby(pts),
        )

    @staticmethod
    def _movement(old_pts, new_pts) -> float:
        if old_pts is None or new_pts is None:
            return 0.0
        try:
            import numpy as np
            return float(np.linalg.norm(new_pts - old_pts))
        except Exception:
            return 0.0

    def explore_seed(self, seed_id: str) -> TrajectoryEvidence:
        from src.agent_tools import navigate_to_seed

        try:
            room_id = int(seed_id)
        except (ValueError, TypeError):
            room_id = 0

        old_pts = self._pts
        pts, angle, success, status, _img = navigate_to_seed(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            room_id,
            self.cfg,
            self.memory,
            self.scene.cam_intrinsic,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
            self._step_counter,
        )
        self._pts, self._angle = pts, angle
        if self._pts is not None and old_pts is not None:
            self._path_length += float(np.linalg.norm(np.asarray(self._pts) - np.asarray(old_pts)))
        self._sync_step_counter()

        arrived_room = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else room_id
        )

        return TrajectoryEvidence(
            subgoal=f"Navigate to seed {seed_id}",
            task_mode="explore_seed",
            progress=f"Arrived at seed {seed_id}, room {arrived_room}",
            salient=[f"seed_{seed_id}", f"room_{arrived_room}"],
            outcome="arrived_near_target" if success else "target_not_reached",
            current_image_b64=_img,
            room_id=arrived_room,
            objects_nearby=self._collect_nearby(pts),
        )

    def explore_frontier(self, frontier_id: str) -> TrajectoryEvidence:
        from src.agent_tools import navigate_to_frontier

        try:
            fid = int(frontier_id)
        except (ValueError, TypeError):
            fid = 0

        old_pts = self._pts
        pts, angle, success, status, _img = navigate_to_frontier(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            fid,
            self.cfg,
            self.memory,
            self.scene.cam_intrinsic,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
            self._step_counter,
        )
        self._pts, self._angle = pts, angle
        if self._pts is not None and old_pts is not None:
            self._path_length += float(np.linalg.norm(np.asarray(self._pts) - np.asarray(old_pts)))
        self._sync_step_counter()

        arrived_room = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )

        return TrajectoryEvidence(
            subgoal=f"Navigate to frontier {frontier_id}",
            task_mode="explore_frontier",
            progress=f"Arrived at frontier {frontier_id}, room {arrived_room}",
            salient=[f"frontier_{frontier_id}", f"room_{arrived_room}"],
            outcome="arrived_near_target" if success else "target_not_reached",
            current_image_b64=_img,
            room_id=arrived_room,
            objects_nearby=self._collect_nearby(pts),
        )

    # ── dispatch ──────────────────────────────────────────────────────

    def execute_action(self, action) -> TrajectoryEvidence:
        if action.action_type == "explore_panorama":
            return self.explore_panorama()
        elif action.action_type == "navigate_to_object":
            return self.navigate_to_object(action.object_name, action.view_idx)
        elif action.action_type == "explore_seed":
            return self.explore_seed(action.seed_id)
        elif action.action_type == "explore_frontier":
            return self.explore_frontier(action.frontier_id)
        elif action.action_type == "submit_answer":
            return TrajectoryEvidence(
                subgoal="Submit answer",
                task_mode="submit_answer",
                progress=f"Answer: {action.answer}",
                salient=[action.answer or ""],
                outcome="answer_submitted",
                room_id=-1,
                objects_nearby=[],
            )
        else:
            logger.warning("Unknown action_type: %s", action.action_type)
            return TrajectoryEvidence(
                subgoal="Unknown action",
                task_mode="unknown",
                progress="Unknown action type",
                outcome="error",
                salient=[],
                gd_quality="no_detection",
            )
