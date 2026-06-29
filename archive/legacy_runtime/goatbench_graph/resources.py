"""GoatbenchResources — heavy object handles injected via RunnableConfig.configurable.

Kept OUT of `GoatbenchState` so state stays serializable/checkpointable. Nodes
access these via `config["configurable"]["resources"]`.

Mirrors the objects constructed in the per-episode setup of
`run_goatbench_evaluation.py` (Scene, TSDFPlanner, cfg, cam_intr, logger,
models, dirs, subtask_metadata).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GoatbenchResources:
    """Heavy resources shared across nodes. Built once by entrypoint."""

    # ── Perception + planning stack ──
    scene: Any                          # src.scene_goatbench.Scene
    tsdf_planner: Any                   # src.tsdf_planner.TSDFPlanner
    cfg: Any                            # OmegaConf cfg
    cam_intr: Any                       # camera intrinsics matrix

    # ── Observability ──
    logger: Any                         # src.logger_goatbench.Logger

    # ── Perception models (dict) ──
    # keys: detection, sam, clip, clip_preprocess, clip_tokenizer
    models: dict

    # ── Output dirs ──
    eps_frontier_dir: str
    eps_snapshot_dir: str
    episode_dir: str

    # ── Subtask metadata (read-only handle; holds goal_obj_ids, viewpoints, etc.) ──
    subtask_metadata: dict
