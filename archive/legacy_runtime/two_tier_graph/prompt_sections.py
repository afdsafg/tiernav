"""Prompt section registry — borrowed from Claude Code's getSystemPrompt().

Static sections (cacheable=True) form a prefix that hits VLM prompt cache.
Dynamic sections (cacheable=False) change every round.

Cache boundary: cacheable sections first (task/schema/memory_index/query),
then non-cacheable (reasoning_history/current_views/topdown). Provider reads
section.cacheable to know where to insert cache_breakpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from src.agent_planner import PLANNER_SYSTEM_PROMPT


@dataclass
class PromptSection:
    name: str
    content: Union[str, list]
    cacheable: bool


def build_planner_prompt(
    state: dict, resources: Optional[Any]
) -> list[PromptSection]:
    """Build ordered prompt sections. Cacheable first, then non-cacheable.

    Cacheable: task_instruction, action_schema, memory_index, active_query.
    Non-cacheable: reasoning_history, current_views, topdown.
    """
    return [
        PromptSection("task_instruction", PLANNER_SYSTEM_PROMPT, cacheable=True),
        PromptSection(
            "action_schema",
            resources.tool_registry.actions_prompt_text() if resources else "",
            cacheable=True,
        ),
        PromptSection(
            "memory_index", state.get("l0_index_text", ""), cacheable=True
        ),
        PromptSection(
            "active_query", state.get("question", ""), cacheable=True
        ),
        PromptSection(
            "reasoning_history", _build_reasoning_history(state), cacheable=False
        ),
        PromptSection(
            "current_views",
            [v.get("image_b64", "") for v in state.get("current_views", [])],
            cacheable=False,
        ),
        PromptSection(
            "topdown", state.get("topdown_b64", ""), cacheable=False
        ),
    ]


def _build_reasoning_history(state: dict) -> str:
    """Summarize round_traces into reasoning history text."""
    traces = state.get("round_traces", [])
    if not traces:
        return ""
    lines = []
    for t in traces:
        if isinstance(t, dict):
            lines.append(
                f"Round {t.get('round_id', '?')}: {t.get('action', '')} — {t.get('reason', '')}"
            )
        else:
            lines.append(str(t))
    return "\n".join(lines)
