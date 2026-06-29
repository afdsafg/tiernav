"""Multi-agent fork stub — borrowed from Claude Code subagent fork pattern.

CacheSafeParams holds cache-safe state snapshot for sidechain subagent.
ForkSubagentTool.run() is a stub (raises NotImplementedError).
Schema + registry + sidechain path structure only.
"""
from dataclasses import dataclass
from typing import Any

from src.two_tier_graph.tools import ActionTool, ToolSchema


@dataclass
class CacheSafeParams:
    """Cache-safe snapshot of planner state for fork subagent.

    Only includes cacheable (static) fields — avoids invalidating prompt cache.
    """
    system_prompt: str
    scene_analysis: str
    action_schema: str


class ForkSubagentTool(ActionTool):
    """Stub for multi-agent fork subagent tool.

    When implemented, will spawn a sidechain subagent with cache-safe params
    to explore alternative action sequences. run() raises NotImplementedError.
    """

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="fork_subagent",
            arg_fields=[("query", "str")],
            prompt_description=(
                '6. fork_subagent: {"reasoning":"why","action":"fork_subagent",'
                '"arguments":{"query":"what to explore"}} (stub)'
            ),
            is_terminal=False,
        )

    def run(self, action: Any = None, ctx: Any = None, *,
            query: str = "", cache_params: CacheSafeParams = None) -> str:
        """Stub — not yet implemented."""
        raise NotImplementedError(
            "ForkSubagentTool.run() is a stub. Implementation pending Phase 3."
        )
