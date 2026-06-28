"""Task adapters that translate benchmark inputs to EpisodeRequest and
EpisodeResult back to eval/logger payloads.

Adapters are the only place benchmark-specific shape leaks into the runtime.
No external services, no LangGraph.
"""
from __future__ import annotations

from typing import Any

from .contracts import EpisodeRequest, EpisodeResult, TaskMode


class AEQATaskAdapter:
    """AEQA: episode = one question-answering turn per scene."""

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
        }


class GOATBenchTaskAdapter:
    """GOATBench: episode = one goal-navigation subtask per scene."""

    task_name = "goatbench"

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
        }
