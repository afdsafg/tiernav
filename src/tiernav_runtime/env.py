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
        self._episode_id: Optional[str] = None
        self._is_torn_down: bool = False

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

    def distance_to_goal(self) -> Optional[float]:
        """Euclidean floor-plane distance from current_pose to goal_pose.

        Habitat coords are [x, y, z] with y up; the floor plane is x, z.
        Returns None when either pose is missing, so the evaluator can
        distinguish "no measurement" from "far from goal".
        """
        if not self._current_pose:
            return None
        if self._goal_pose is None:
            return None
        dx = self._current_pose.get("x", 0.0) - self._goal_pose.get("x", 0.0)
        dz = self._current_pose.get("z", 0.0) - self._goal_pose.get("z", 0.0)
        return (dx * dx + dz * dz) ** 0.5

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
        self._current_pose = dict(initial_pose)
        self._path_length = 0.0
        if self.executor is not None and hasattr(self.executor, "set_state"):
            # Habitat uses 3D pts [x, y, z] where y is up. The pose dict
            # carries x, z (floor plane) and optional y (floor height).
            # Fall back to 0.0 for y when unset (legacy 2D pose).
            x = initial_pose.get("x", 0.0)
            y = initial_pose.get("y", 0.0)
            z = initial_pose.get("z", 0.0)
            pts = [x, y, z]
            angle = initial_pose.get("theta", 0.0)
            try:
                self.executor.set_state(pts, angle, 0)
            except TypeError:
                # Executor signature mismatch — leave pose bookkeeping to env.
                pass

    def teardown_session(self) -> None:
        """Clean up scene resources. Idempotent."""
        if self._is_torn_down:
            return
        if self.scene is not None and hasattr(self.scene, "cleanup"):
            self.scene.cleanup()
        self._is_torn_down = True
