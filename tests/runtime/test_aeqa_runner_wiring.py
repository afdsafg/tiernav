import numpy as np
from pathlib import Path

from src.tiernav_runtime.pose_utils import initial_pose_from_pts


def test_initial_pose_from_3d_pts_preserves_z_axis():
    pose = initial_pose_from_pts(np.array([1.0, 2.0, 3.0]), 0.75)
    assert pose == {"x": 1.0, "y": 2.0, "z": 3.0, "theta": 0.75}


def test_initial_pose_from_2d_pts_defaults_z_to_zero():
    pose = initial_pose_from_pts([4.0, 5.0], 1.25)
    assert pose == {"x": 4.0, "y": 5.0, "z": 0.0, "theta": 1.25}


def test_aeqa_runner_wires_predictive_controller_and_tool_registry():
    source = Path("run_two_tier_aeqa_evaluation.py").read_text(encoding="utf-8")

    assert "AEQAPredictiveController" in source
    assert "build_aeqa_tool_registry" in source
    assert "tools=aeqa_tools" in source
    assert "aeqa_controller=aeqa_controller" in source
    assert "initial_pose = initial_pose_from_pts(start_pts, start_angle)" in source
