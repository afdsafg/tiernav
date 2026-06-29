"""Resources — heavy/object handles injected via RunnableConfig.configurable.

Kept OUT of `TwoTierState` so state stays serializable/checkpointable. Nodes
access these via `config["configurable"]["resources"]`.

Mirrors the objects constructed in `run_episode_two_tier` (agent_workflow.py:1150-1221):
Scene, TSDFPlanner, MemoryStore, Executor, Planner, EvidenceNotebook,
SceneGraphMemory, plus the new abstractions (LLMProvider, ToolRegistry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Resources:
    """Heavy resources shared across nodes. Built once by the entrypoint."""

    # ── Perception stack (built in init_node) ──
    scene: Any                               # src.scene_aeqa.Scene
    tsdf_planner: Any                        # src.tsdf_planner.TSDFPlanner
    memory_store: Any                        # src.agent_memory.MemoryStore
    models: dict                             # detection / sam / clip / clip_preprocess / clip_tokenizer
    cfg: Any

    # ── Coexisting memory stores (handles; summaries mirrored into state) ──
    notebook: Any                            # src.agent_notebook.EvidenceNotebook
    scene_graph: Optional[Any]               # src.scene_graph_memory.SceneGraphMemory | None

    # ── Agent components ──
    planner: Any                             # src.agent_planner.Planner (used by MimoProvider)
    executor: Any                            # src.agent_executor.Executor
    llm_provider: Any                        # LLMProvider (see providers.py)
    tool_registry: Any                       # ToolRegistry (see tools.py)

    # ── Observability ──
    run_logger: Optional[Any]                # src.run_logger.RunLogger | None

    # ── Episode identity (for run_logger calls) ──
    question_id: str = ""
    question: str = ""
    output_dir: str = ""

    # ── GOATBench 任务上下文（AEQA 路径为 None）──
    goal_type: Optional[str] = None          # "object"|"description"|"image"|None
    goal_metadata: Optional[dict] = None     # {"goal_description": str, ...}（不含真值位置）
