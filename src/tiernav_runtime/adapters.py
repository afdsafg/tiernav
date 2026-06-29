"""Task adapters that translate benchmark inputs to EpisodeRequest and
EpisodeResult back to eval/logger payloads.

Adapters are the only place benchmark-specific shape leaks into the runtime.
No external services, no LangGraph.
"""
from __future__ import annotations

from typing import Any, Optional

from .contracts import EpisodeRequest, EpisodeResult, TaskMode


class AEQATaskAdapter:
    """AEQA: episode = one question-answering turn per scene.

    Each question is independent — no cross-question memory.  The adapter is a
    pure data mapper with no session state.
    """

    task_name = "aeqa"

    def to_request(
        self,
        scene_id: str,
        question_id: str,
        question: str,
        output_dir: str,
        initial_pose: dict[str, float] | None = None,
    ) -> EpisodeRequest:
        return EpisodeRequest(
            episode_id=question_id,
            scene_id=scene_id,
            task_name=self.task_name,
            task_mode=TaskMode.QUESTION_ANSWERING,
            prompt=question,
            initial_pose=initial_pose or {},
            output_dir=output_dir,
        )

    def to_eval_payload(self, result: EpisodeResult) -> dict[str, Any]:
        return {
            "question_id": result.episode_id,
            "scene_id": result.scene_id,
            "answer": result.answer,
            "success": result.success,
            "steps_taken": result.steps_taken,
            "rounds_used": result.rounds_used,
            "path_length": result.path_length,
            "error": result.error,
            "distance_to_goal": result.distance_to_goal,
            "submit_was_explicit": result.submit_was_explicit,
        }


class GOATBenchTaskAdapter:
    """GOATBench: episode = a sequence of goal-navigation subtasks.

    **Session threading** (runtime path): ``start_episode`` opens a long-lived
    episode session; ``run_subtask`` produces ``EpisodeRequest`` objects whose
    ``episode_id`` is the per-episode id (not a per-subtask composite), so
    downstream memory services see the same session across subtasks.

    **Legacy path**: ``to_request`` still produces per-subtask ids for
    backward compatibility with non-runtime runners.
    """

    task_name = "goatbench"

    def __init__(self) -> None:
        self._episode_id: Optional[str] = None
        self._scene_id: str = ""
        self._output_dir: str = ""

    # -- Runtime session API ------------------------------------------------

    def start_episode(
        self,
        episode_id: str,
        *,
        scene_id: str = "",
        output_dir: str = "",
    ) -> None:
        """Open a long-lived episode session for subtask threading."""
        self._episode_id = episode_id
        self._scene_id = scene_id
        self._output_dir = output_dir

    def run_subtask(
        self,
        subtask_index: int,
        goal_type: str,
        goal_description: str,
        *,
        initial_pose: dict[str, float] | None = None,
    ) -> EpisodeRequest:
        """Produce an EpisodeRequest sharing the episode-level id.

        Raises RuntimeError if ``start_episode`` has not been called.
        """
        if self._episode_id is None:
            raise RuntimeError(
                "GOATBenchTaskAdapter.start_episode must be called before run_subtask"
            )
        return EpisodeRequest(
            episode_id=self._episode_id,
            scene_id=self._scene_id,
            task_name=self.task_name,
            task_mode=TaskMode.GOAL_NAVIGATION,
            prompt=f"Navigate to {goal_description}",
            goal_metadata={
                "goal_type": goal_type,
                "goal_description": goal_description,
                "subtask_index": subtask_index,
            },
            initial_pose=initial_pose or {},
            output_dir=self._output_dir,
        )

    # -- Legacy API ---------------------------------------------------------

    def to_request(
        self,
        scene_id: str,
        episode_id: str,
        subtask_index: int,
        goal_type: str,
        goal_description: str,
        output_dir: str,
        initial_pose: dict[str, float] | None = None,
    ) -> EpisodeRequest:
        """Legacy per-subtask composite id (backward compat)."""
        subtask_id = f"{episode_id}_{subtask_index}"
        return EpisodeRequest(
            episode_id=subtask_id,
            scene_id=scene_id,
            task_name=self.task_name,
            task_mode=TaskMode.GOAL_NAVIGATION,
            prompt=f"Navigate to {goal_description}",
            goal_metadata={
                "goal_type": goal_type,
                "goal_description": goal_description,
                "subtask_index": subtask_index,
            },
            initial_pose=initial_pose or {},
            output_dir=output_dir,
        )

    # -- Shared -------------------------------------------------------------

    def to_eval_payload(self, result: EpisodeResult) -> dict[str, Any]:
        return {
            "subtask_id": result.episode_id,
            "scene_id": result.scene_id,
            "success": result.success,
            "answer": result.answer,
            "steps_taken": result.steps_taken,
            "rounds_used": result.rounds_used,
            "path_length": result.path_length,
            "failure_type": result.failure_type,
            "error": result.error,
            "distance_to_goal": result.distance_to_goal,
            "submit_was_explicit": result.submit_was_explicit,
        }
