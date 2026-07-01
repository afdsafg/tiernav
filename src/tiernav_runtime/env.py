"""Runtime environment service: owns Habitat-backed heavy objects and session state.

This is an OWNERSHIP and SESSION-SEMANTICS layer, not a construction-from-
scratch layer. The caller (runner/adapter, Task 8/9) constructs the Scene,
TSDFPlanner, models, and Executor — typically from habitat_sim — and hands
them to :meth:`RuntimeEnvironmentService.for_aeqa` /
:meth:`RuntimeEnvironmentService.for_goatbench`. The service then owns them
for the lifetime of one AEQA question or one GOATBench episode and exposes
``current_pose`` / ``path_length`` plus session lifecycle.

Session semantics:
- AEQA (``question_answering``): fresh session per question.
  ``start_session`` resets pose and path_length.
- GOATBench (``goal_navigation``): long-lived session per episode.
  ``start_session`` threads pose/path_length across subtasks (only the first
  subtask in an episode seeds initial pose; later subtasks preserve state).

The graph is NOT wired to this service here; Task 7 does that. This service
is constructed and made available to the runtime so production code can build
:class:`RuntimeServices` from a real environment instead of the fake defaults.
"""
from __future__ import annotations

from logging import Logger
from typing import Any, Optional

from .contracts import TaskMode


class RuntimeEnvironmentService:
    """Owns scene/planner/models/pose for one AEQA question or GOATBench episode.

    Heavy objects (Scene, TSDFPlanner, Executor, detection/SAM/CLIP models)
    are injected fully constructed. This keeps the service deterministic in
    tests: no habitat_sim import, no GPU.
    """

    def __init__(
        self,
        *,
        task_mode: TaskMode,
        scene: Any,
        tsdf_planner: Any,
        executor: Any,
        detection_model: Any = None,
        sam_predictor: Any = None,
        clip_model: Any = None,
        clip_preprocess: Any = None,
        clip_tokenizer: Any = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self._task_mode = task_mode
        self.scene = scene
        self.tsdf_planner = tsdf_planner
        self.executor = executor
        self.detection_model = detection_model
        self.sam_predictor = sam_predictor
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.logger = logger

        self._current_pose: dict[str, float] = {}
        self._path_length: float = 0.0
        self._goal_pose: Optional[dict[str, float]] = None
        self._goal_poses: list[dict[str, float]] = []
        self._episode_id: Optional[str] = None
        self._is_torn_down: bool = False
        self._initial_visual_observation: Optional[Any] = None

    # --- Properties -------------------------------------------------------

    @property
    def task_mode(self) -> str:
        return self._task_mode.value

    @property
    def current_pose(self) -> dict[str, float]:
        return dict(self._current_pose)

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def goal_pose(self) -> Optional[dict[str, float]]:
        return dict(self._goal_pose) if self._goal_pose is not None else None

    def set_goal_pose(self, pose: Optional[dict[str, float]]) -> None:
        """Set the goal pose for distance computation.

        Set during session start from EpisodeRequest.goal_metadata. Tests
        may set a fake goal_pose directly.
        """
        self._goal_pose = dict(pose) if pose is not None else None
        self._goal_poses = [self._goal_pose] if self._goal_pose is not None else []

    def set_goal_poses(self, poses: list[dict[str, float]]) -> None:
        """Set multiple goal viewpoints for MultiGoal distance computation.

        GOATBench objects have multiple view_points; success is measured as
        the shortest distance to ANY of them. Replaces the single-pose path.
        """
        self._goal_poses = [dict(p) for p in poses] if poses else []
        self._goal_pose = self._goal_poses[0] if self._goal_poses else None

    def distance_to_goal(self) -> Optional[float]:
        """Euclidean floor-plane distance to the NEAREST goal viewpoint.

        Habitat coords are [x, y, z] with y up; the floor plane is x, z.
        Returns None when either pose is missing, so the evaluator can
        distinguish "no measurement" from "far from goal".
        """
        if not self._current_pose:
            return None
        if not self._goal_poses:
            return None
        cx = self._current_pose.get("x", 0.0)
        cz = self._current_pose.get("z", 0.0)
        best = None
        for gp in self._goal_poses:
            dx = cx - gp.get("x", 0.0)
            dz = cz - gp.get("z", 0.0)
            d = (dx * dx + dz * dz) ** 0.5
            if best is None or d < best:
                best = d
        return best

    @property
    def is_torn_down(self) -> bool:
        return self._is_torn_down

    # --- Factories -------------------------------------------------------

    @classmethod
    def for_aeqa(
        cls,
        *,
        scene: Any,
        tsdf_planner: Any,
        executor: Any,
        detection_model: Any = None,
        sam_predictor: Any = None,
        clip_model: Any = None,
        clip_preprocess: Any = None,
        clip_tokenizer: Any = None,
        logger: Optional[Logger] = None,
    ) -> "RuntimeEnvironmentService":
        """Build a service for one AEQA question (fresh session per question)."""
        return cls(
            task_mode=TaskMode.QUESTION_ANSWERING,
            scene=scene,
            tsdf_planner=tsdf_planner,
            executor=executor,
            detection_model=detection_model,
            sam_predictor=sam_predictor,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
            logger=logger,
        )

    @classmethod
    def for_goatbench(
        cls,
        *,
        scene: Any,
        tsdf_planner: Any,
        executor: Any,
        detection_model: Any = None,
        sam_predictor: Any = None,
        clip_model: Any = None,
        clip_preprocess: Any = None,
        clip_tokenizer: Any = None,
        logger: Optional[Logger] = None,
    ) -> "RuntimeEnvironmentService":
        """Build a service for one GOATBench episode (long-lived, pose threading)."""
        return cls(
            task_mode=TaskMode.GOAL_NAVIGATION,
            scene=scene,
            tsdf_planner=tsdf_planner,
            executor=executor,
            detection_model=detection_model,
            sam_predictor=sam_predictor,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
            logger=logger,
        )

    # --- Session lifecycle -----------------------------------------------

    def start_session(
        self,
        episode_id: str,
        *,
        initial_pose: Optional[dict[str, float]] = None,
    ) -> None:
        """Begin (or continue) a session.

        AEQA: every call resets pose/path_length — each question is independent.
        GOATBench: only the first call of a new episode seeds state; later
        same-episode calls thread pose/path_length from where the prior
        subtask ended.
        """
        self._is_torn_down = False

        if self._task_mode is TaskMode.QUESTION_ANSWERING:
            self._reset_session(initial_pose or {})
        else:
            # GOATBench: thread across subtasks within one episode; reset
            # only when entering a fresh episode.
            if episode_id != self._episode_id:
                self._reset_session(initial_pose or {})

        self._episode_id = episode_id

    def _reset_session(self, initial_pose: dict[str, float]) -> None:
        import numpy as np

        self._current_pose = dict(initial_pose)
        self._path_length = 0.0
        self._initial_visual_observation = None
        if self.executor is not None and hasattr(self.executor, "set_state"):
            # Habitat uses 3D pts [x, y, z] where y is up. numpy array so
            # downstream code can do pts[[0, 2]] fancy indexing.
            x = initial_pose.get("x", 0.0)
            y = initial_pose.get("y", 0.0)
            z = initial_pose.get("z", 0.0)
            pts = np.array([x, y, z], dtype=np.float32)
            angle = initial_pose.get("theta", 0.0)
            try:
                self.executor.set_state(pts, angle, 0)
            except TypeError:
                # Executor signature mismatch — leave pose bookkeeping to env.
                pass

    # --- AEQA visual adapter ---------------------------------------------

    def initialize_aeqa_visual_context(self) -> None:
        """Run the fixed AEQA initial panorama to seed snapshots and frontiers.

        The AEQA planner only sees `explore_frontier` and `submit_answer`, but
        Pred-EQA-style first-step decisions need visual evidence. The existing
        executor's panorama path performs real perception, snapshot storage, and
        frontier construction; this method exposes it as environment
        initialization rather than as a VLM-selectable tool.
        """
        if self._task_mode is not TaskMode.QUESTION_ANSWERING:
            return
        if self.executor is None or not hasattr(self.executor, "explore_panorama"):
            return
        self._initial_visual_observation = self.executor.explore_panorama()
        self._refresh_frontier_map_from_executor()
        self._sync_pose_from_executor()

    def _refresh_frontier_map_from_executor(self) -> None:
        tsdf = self.tsdf_planner
        executor = self.executor
        if tsdf is None or executor is None or not hasattr(tsdf, "update_frontier_map"):
            return
        pts = getattr(executor, "_pts", None)
        if pts is None:
            return
        cfg = getattr(getattr(executor, "cfg", None), "planner", None)
        if cfg is None:
            return
        try:
            tsdf.update_frontier_map(
                pts,
                cfg,
                self.scene,
                int(getattr(executor, "_step_counter", 0) or 0),
                save_frontier_image=False,
            )
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("AEQA initial frontier refresh failed: %s", exc)

    def get_aeqa_visual_state(self, episode: Any) -> dict[str, Any]:
        """Return real image evidence for the AEQA predictive controller."""
        return {
            "question": str(getattr(episode, "prompt", "") or ""),
            "current_step": int(getattr(episode, "step_index", 0) or 0),
            "snapshots": self._build_aeqa_snapshots(),
            "frontiers": self._build_aeqa_frontiers(),
            "egocentric_views": self._build_aeqa_egocentric_views(episode),
            "memory_text": self._build_aeqa_memory_text(),
            "tool_feedback": self._build_aeqa_tool_feedback(episode),
        }

    def _sync_pose_from_executor(self) -> None:
        executor = self.executor
        if executor is None or not hasattr(executor, "_pts") or executor._pts is None:
            return
        pts = executor._pts
        angle = getattr(executor, "_angle", 0.0) or 0.0
        self._current_pose = {
            "x": float(pts[0]) if len(pts) > 0 else 0.0,
            "y": float(pts[1]) if len(pts) > 1 else 0.0,
            "z": float(pts[2]) if len(pts) > 2 else 0.0,
            "theta": float(angle),
        }
        self._path_length = float(getattr(executor, "_path_length", 0.0) or 0.0)

    @staticmethod
    def _image_to_b64(image: Any) -> str:
        if image is None:
            return ""
        if isinstance(image, str):
            if image.startswith("data:image/") and "base64," in image:
                return image.split("base64,", 1)[1]
            return image
        try:
            from src.agent_image_utils import numpy_to_base64
            import numpy as np

            arr = np.asarray(image)
            if arr.ndim < 2:
                return ""
            if arr.ndim == 3 and arr.shape[-1] > 3:
                arr = arr[..., :3]
            return numpy_to_base64(arr, fmt="PNG")
        except Exception:
            return ""

    def _build_aeqa_snapshots(self) -> list[dict[str, str]]:
        scene = self.scene
        snapshots = getattr(scene, "snapshots", {}) or {}
        observations = getattr(scene, "all_observations", {}) or {}
        objects = getattr(scene, "objects", {}) or {}
        items = snapshots.items() if hasattr(snapshots, "items") else enumerate(snapshots)

        result: list[dict[str, str]] = []
        for idx, (snapshot_id, snapshot) in enumerate(items):
            image = observations.get(snapshot_id)
            if image is None:
                image = getattr(snapshot, "image", None)
            image_b64 = self._image_to_b64(image)
            if not image_b64:
                continue
            classes = self._snapshot_class_names(snapshot, objects)
            label = f"Snapshot {idx}"
            if classes:
                label += " objects: " + ", ".join(classes)
            result.append({
                "image_id": str(snapshot_id),
                "image_b64": image_b64,
                "label": label,
                "source": "snapshot",
            })
        return result

    @staticmethod
    def _snapshot_class_names(snapshot: Any, objects: Any) -> list[str]:
        names: list[str] = []
        for obj_id in getattr(snapshot, "cluster", []) or []:
            obj = None
            try:
                obj = objects.get(obj_id)
            except Exception:
                obj = None
            if obj is None:
                try:
                    obj = objects.get(str(obj_id))
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                name = obj.get("class_name") or obj.get("name")
                if name:
                    names.append(str(name))
        return sorted(set(names))

    def _build_aeqa_frontiers(self) -> list[dict[str, str]]:
        frontiers = getattr(self.tsdf_planner, "frontiers", []) or []
        result: list[dict[str, str]] = []
        for frontier in frontiers:
            image_b64 = self._image_to_b64(getattr(frontier, "feature", None))
            if not image_b64:
                continue
            frontier_id = str(getattr(frontier, "frontier_id", len(result)))
            label = f"Frontier {frontier_id}"
            room_id = getattr(frontier, "room_id", None)
            if room_id is not None:
                label += f" room {room_id}"
            result.append({
                "frontier_id": frontier_id,
                "image_b64": image_b64,
                "label": label,
            })
        return result

    def _build_aeqa_egocentric_views(self, episode: Any) -> list[dict[str, str]]:
        views: list[dict[str, str]] = []
        initial_b64 = self._image_to_b64(
            getattr(self._initial_visual_observation, "current_image_b64", None)
        )
        if initial_b64:
            views.append({
                "image_id": "initial_panorama",
                "image_b64": initial_b64,
                "label": "Initial panorama",
                "source": "egocentric",
            })

        last_observation = getattr(episode, "last_observation", None)
        raw = getattr(last_observation, "raw", {}) if last_observation is not None else {}
        current_b64 = ""
        if isinstance(raw, dict):
            current_b64 = self._image_to_b64(raw.get("current_image_b64"))
        if current_b64:
            views.append({
                "image_id": "current_view",
                "image_b64": current_b64,
                "label": "Current egocentric view",
                "source": "egocentric",
            })
        return views

    def _build_aeqa_memory_text(self) -> str:
        lines: list[str] = []
        if self._initial_visual_observation is not None:
            progress = getattr(self._initial_visual_observation, "progress", "") or ""
            if progress:
                lines.append("Initial observation: " + str(progress))
        frontiers = getattr(self.tsdf_planner, "frontiers", []) or []
        if frontiers:
            ids = [str(getattr(frontier, "frontier_id", "?")) for frontier in frontiers[:20]]
            lines.append("Available frontiers: " + ", ".join(ids))
        return "\n".join(lines)

    def _build_aeqa_tool_feedback(self, episode: Any) -> str:
        last_observation = getattr(episode, "last_observation", None)
        summary = getattr(last_observation, "summary", "") if last_observation is not None else ""
        if summary:
            return str(summary)
        if self._initial_visual_observation is not None:
            return str(getattr(self._initial_visual_observation, "progress", "") or "")
        return ""

    def teardown_session(self) -> None:
        """Clean up scene resources. Idempotent."""
        if self._is_torn_down:
            return
        if self.scene is not None and hasattr(self.scene, "cleanup"):
            self.scene.cleanup()
        self._is_torn_down = True
