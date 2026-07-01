"""Small pose helpers shared by runtime runners and tests."""
from __future__ import annotations


def initial_pose_from_pts(start_pts, start_angle: float) -> dict[str, float]:
    pts_array = getattr(start_pts, "tolist", None)
    if pts_array is not None:
        pts_list = pts_array()
    else:
        pts_list = list(start_pts) if start_pts is not None else [0.0, 0.0, 0.0]
    x = float(pts_list[0]) if len(pts_list) > 0 else 0.0
    y = float(pts_list[1]) if len(pts_list) > 1 else 0.0
    z = float(pts_list[2]) if len(pts_list) > 2 else 0.0
    return {
        "x": x,
        "y": y,
        "z": z,
        "theta": float(start_angle),
    }
