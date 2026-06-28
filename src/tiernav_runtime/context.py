"""Sectioned context compiler for the TierNav runtime.

Compiles an :class:`EpisodeState` into an ordered list of
:class:`ContextSection` objects with explicit cacheable/dynamic boundaries,
suitable for prompt-caching-aware model providers (Claude Code style).

The compiler is contract-first: it depends only on stdlib and the runtime
contracts module. No external services, no LangGraph, no fabricated evidence.
"""
from __future__ import annotations

import hashlib
from typing import Any

from src.tiernav_runtime.contracts import ContextSection, EpisodeState


def _hash(content: str) -> str:
    """Return sha256 hex digest of ``content`` (utf-8), or "" for empty content."""
    if not content:
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _estimate_tokens(content: str) -> int:
    """Cheap, deterministic token estimate.

    Empty content -> 0. Non-empty -> positive int. Uses a simple
    whitespace-plus-punctuation split so identical content always yields the
    same count without depending on a tokenizer. Good enough for cache-break
    decisions and budget accounting; the real planner may re-estimate.
    """
    if not content:
        return 0
    # ponytail: ceiling upgrade path — swap for a real tokenizer when available.
    return max(1, len(content.split()))


def _section(name: str, content: str, cacheable: bool) -> ContextSection:
    return ContextSection(
        name=name,
        content=content,
        cacheable=cacheable,
        token_estimate=_estimate_tokens(content),
        content_hash=_hash(content),
    )


def _format_memory_pack(pack: Any) -> str:
    """Render a MemoryPack's cacheable summary fields as a compact block."""
    lines: list[str] = []
    if pack.summary:
        lines.append(f"summary: {pack.summary}")
    if pack.evidence_ids:
        lines.append("evidence_ids: " + ", ".join(pack.evidence_ids))
    if pack.reuse_hint:
        lines.append(f"reuse_hint: {pack.reuse_hint}")
    return "\n".join(lines)


def render_prompt(sections: list[ContextSection]) -> str:
    """Render non-empty sections into model-readable text.

    Each non-empty section is emitted under a markdown-style header. Empty
    sections are skipped entirely (no orphan headers), which keeps the prompt
    stable across cache/dynamic boundary changes.
    """
    blocks: list[str] = []
    for section in sections:
        if not section.content:
            continue
        blocks.append(f"## {section.name}\n{section.content}")
    return "\n\n".join(blocks)


class ContextCompiler:
    """Compile :class:`EpisodeState` into ordered, cache-annotated sections.

    Section order is stable and groups cacheable content first:

      1. task_instruction      (cacheable)  — prompt + task identity
      2. action_schema         (cacheable)  — available actions
      3. memory_index          (cacheable)  — memory pack summary
      4. recent_trace          (dynamic)    — round/step counters
      5. current_observation   (dynamic)    — latest observation summary
      6. policy_hint           (dynamic)    — optional steering hint

    The cacheable prefix (1-3) is stable for identical inputs, enabling
    prompt-cache reuse. Dynamic sections (4-6) change every step.
    """

    def compile(
        self,
        state: EpisodeState,
        action_schema: str,
        include_memory: bool = True,
        policy_hint: str = "",
    ) -> list[ContextSection]:
        task_instruction = self._render_task_instruction(state)
        memory_text = self._render_memory(state, include_memory)
        recent_trace = self._render_recent_trace(state)
        observation_text = self._render_observation(state)
        policy_text = policy_hint

        return [
            _section("task_instruction", task_instruction, cacheable=True),
            _section("action_schema", action_schema, cacheable=True),
            _section("memory_index", memory_text, cacheable=True),
            _section("recent_trace", recent_trace, cacheable=False),
            _section("current_observation", observation_text, cacheable=False),
            _section("policy_hint", policy_text, cacheable=False),
        ]

    def render_prompt(self, sections: list[ContextSection]) -> str:
        """Render sections to a model-facing prompt.

        Instance method delegating to the module-level :func:`render_prompt`
        so callers may use the planned ``compiler.render_prompt(compiler.compile(...))``
        form. The module-level function is retained for backwards compatibility.
        """
        return render_prompt(sections)

    @staticmethod
    def _render_task_instruction(state: EpisodeState) -> str:
        lines = [
            f"episode_id: {state.episode_id}",
            f"scene_id: {state.scene_id}",
            f"task_name: {state.task_name}",
            f"task_mode: {state.task_mode.value}",
            f"prompt: {state.prompt}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_memory(state: EpisodeState, include_memory: bool) -> str:
        if not include_memory:
            return ""
        pack = state.memory_pack
        if pack is None:
            return ""
        return _format_memory_pack(pack)

    @staticmethod
    def _render_recent_trace(state: EpisodeState) -> str:
        return f"round_index: {state.round_index}\nstep_index: {state.step_index}"

    @staticmethod
    def _render_observation(state: EpisodeState) -> str:
        return state.last_observation.summary
