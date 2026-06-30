"""GOATBench runner feeds EpisodeResult + executor state into legacy Logger."""
from unittest.mock import MagicMock
import numpy as np

from src.tiernav_runtime.contracts import (
    EpisodeResult, TaskMode, PlannerDecision,
)


def test_log_subtask_result_called_with_runtime_outputs():
    """After each subtask, logger.log_subtask_result gets success/distance/snapshots."""
    from run_goatbench_evaluation import _feed_result_to_logger

    logger = MagicMock()
    logger.subtask_explore_dist = 2.0
    logger.init_subtask.return_value = {"gt_subtask_explore_dist": 1.5}
    logger.success_by_snapshot = {}
    logger.success_by_distance = {}
    logger.spl_by_snapshot = {}
    logger.spl_by_distance = {}
    logger.success_by_task = {}
    logger.spl_by_task = {}
    logger.n_filtered_snapshots_list = {}
    logger.n_total_snapshots_list = {}
    logger.n_total_frames_list = {}

    result = EpisodeResult(
        episode_id="ep1_0", scene_id="s", task_name="goatbench",
        task_mode=TaskMode.GOAL_NAVIGATION, success=True,
        distance_to_goal=0.5, submit_was_explicit=True,
        path_length=3.0, steps_taken=5,
    )
    executor = MagicMock()
    executor._pts = np.array([1.0, 2.0])
    executor._path_length = 3.0
    scene = MagicMock()
    scene.snapshots = {"a": 1, "b": 2}
    scene.frames = list(range(10))
    tsdf_planner = MagicMock()

    _feed_result_to_logger(
        logger=logger, result=result, executor=executor, scene=scene,
        tsdf_planner=tsdf_planner, subtask_id="ep1_0",
        goal_type="object", subtask_goal=[{
            "object_category": "chair", "object_id": "obj_0",
            "position": [1, 1, 1],
            "view_points": [{"agent_state": {"position": [1, 1, 1]}}],
        }],
        floor_height=0.5,
    )

    logger.init_subtask.assert_called_once()
    logger.log_subtask_result.assert_called_once()
    call = logger.log_subtask_result.call_args.kwargs
    assert call["success_by_distance"] is True
    assert call["subtask_id"] == "ep1_0"
    assert call["gt_subtask_explore_dist"] > 0
