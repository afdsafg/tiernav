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

from .contracts import ContextSection, EpisodeState


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
        env: Any = None,
    ) -> list[ContextSection]:
        if not isinstance(action_schema, str):
            raise TypeError(
                f"action_schema must be str, got {type(action_schema).__name__}: "
                f"{action_schema!r}"
            )
        if not isinstance(policy_hint, str):
            raise TypeError(
                f"policy_hint must be str, got {type(policy_hint).__name__}: "
                f"{policy_hint!r}"
            )
        task_instruction = self._render_task_instruction(state)
        memory_text = self._render_memory(state, include_memory)
        task_state = self._render_task_state(state)
        recent_trace = self._render_recent_trace(state)
        observation_text = self._render_observation(state)
        scene_graph_text = self._render_scene_graph_memory(env, include_memory)
        targets_text = self._render_available_targets(env)
        tool_feedback = self._render_tool_feedback(state)
        policy_text = policy_hint

        return [
            _section("task_instruction", task_instruction, cacheable=True),
            _section("action_schema", action_schema, cacheable=True),
            _section("memory_index", memory_text, cacheable=True),
            _section("task_state", task_state, cacheable=False),
            _section("recent_trace", recent_trace, cacheable=False),
            _section("current_observation", observation_text, cacheable=False),
            _section("scene_graph_memory", scene_graph_text, cacheable=False),
            _section("available_targets", targets_text, cacheable=False),
            _section("tool_feedback", tool_feedback, cacheable=False),
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
            "",
            "You are a navigation planner. Output ONLY a JSON object on a single line, no markdown fences, no prose.",
            "Required fields: action_type (one of the available tools), reason (string), expected (string).",
            "Optional fields: object_name (str), seed_id (str), frontier_id (str), view_idx (int), answer (str, required for submit_answer).",
            "Pick frontier_id / seed_id / object_name from the available_targets section below. Do NOT invent ids.",
            "Do not call explore_frontier when frontiers is none or absent.",
            "Do not call explore_seed when seeds is none or absent.",
            "Do not call navigate_to_object when objects is none or absent.",
            "Strategy: explore_panorama to observe -> explore_frontier/explore_seed to move -> navigate_to_object once target visible -> submit_answer when done.",
            'Example: {"action_type": "explore_panorama", "reason": "Need to observe surroundings", "expected": "Get room layout"}',
            'For target tools, copy the exact frontier_id, seed_id, or object_name from available_targets.',
            'Example: {"action_type": "submit_answer", "reason": "Final answer", "expected": "Done", "answer": "<your answer here>"}',
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
    def _render_task_state(state: EpisodeState) -> str:
        lines = [
            "continuous_context: enabled",
            f"task_mode: {state.task_mode.value}",
            f"round_index: {state.round_index}",
            f"step_index: {state.step_index}",
            "workflow: observe -> choose a valid target -> move -> update memory -> submit only when supported.",
            "Do not repeat failed actions with the same target; use tool_feedback and scene_graph_memory to change strategy.",
        ]
        if state.failure_type:
            lines.append(f"last_failure_type: {state.failure_type}")
        if state.distance_to_goal is not None:
            lines.append(f"distance_to_goal_m: {state.distance_to_goal:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _render_recent_trace(state: EpisodeState) -> str:
        return f"round_index: {state.round_index}\nstep_index: {state.step_index}"

    @staticmethod
    def _render_observation(state: EpisodeState) -> str:
        obs = state.last_observation
        parts: list[str] = []
        if obs.summary:
            parts.append(obs.summary)
        if obs.object_ids:
            parts.append("objects_nearby: " + ", ".join(obs.object_ids))
        if obs.room_id is not None:
            parts.append(f"room_id: {obs.room_id}")
        return "\n".join(parts)

    @staticmethod
    def _render_scene_graph_memory(env: Any, include_memory: bool) -> str:
        if not include_memory or env is None:
            return ""
        graph = getattr(env, "scene_graph_memory", None)
        if graph is None:
            session = getattr(env, "memory_session", None)
            graph = getattr(session, "scene_graph", None) if session is not None else None
        if graph is None or not hasattr(graph, "get_summary_for_planner"):
            return ""
        try:
            return str(graph.get_summary_for_planner())
        except Exception:
            return ""

    @staticmethod
    def _render_tool_feedback(state: EpisodeState) -> str:
        obs = state.last_observation
        raw = obs.raw or {}
        outcome = str(raw.get("outcome", "") or "")
        try:
            path_length = float(raw.get("path_length", 0.0) or 0.0)
        except (TypeError, ValueError):
            path_length = 0.0
        try:
            path_delta = float(raw.get("path_delta", path_length) or 0.0)
        except (TypeError, ValueError):
            path_delta = 0.0
        action = (
            state.current_decision.action_type
            if state.current_decision is not None
            else str(raw.get("action_type", "") or "")
        )
        is_stationary_panorama = (
            action == "explore_panorama"
            and outcome == "panorama_complete"
            and state.step_index > 0
            and path_delta <= 0.0
        )
        if (
            outcome not in {"target_not_reached", "detection_failed", "error"}
            and not is_stationary_panorama
        ):
            return ""

        parts = []
        if action:
            parts.append(f"last_tool_action: {action}")
        parts.append(f"last_tool_outcome: {outcome}")
        if is_stationary_panorama:
            parts.append(
                "guidance: panorama already refreshed context without moving; choose a valid available target and move."
            )
        else:
            parts.append(
                "guidance: the last tool did not make useful progress; try a different target or observe before retrying."
            )
        subgoal = raw.get("subgoal")
        progress = raw.get("progress") or obs.summary
        if subgoal:
            parts.append(f"last_subgoal: {subgoal}")
        if progress:
            parts.append(f"last_progress: {progress}")
        if obs.object_ids:
            parts.append("objects_nearby: " + ", ".join(obs.object_ids))
        return "\n".join(parts)

    @staticmethod
    def _render_available_targets(env: Any) -> str:
        """List frontiers/seeds/objects the planner can target this round.

        Reads from the RuntimeEnvironmentService's tsdf_planner and scene.
        Returns "" when env is None or has no targets, so the section is
        skipped by render_prompt.
        """
        if env is None:
            return ""
        lines: list[str] = []

        # Frontiers — unexplored boundary regions the agent can navigate to.
        tsdf = getattr(env, "tsdf_planner", None)
        frontiers = getattr(tsdf, "frontiers", None) if tsdf is not None else None
        scene = getattr(env, "scene", None)
        _objs = getattr(scene, "objects", None) if scene is not None else None
        if tsdf is not None and frontiers:
            ids = [str(getattr(f, "frontier_id", "?")) for f in frontiers[:20]]
            lines.append("frontiers: " + ", ".join(ids))
        elif tsdf is not None:
            lines.append("frontiers: none")

        # Seeds — room-entry points registered by SeedViewManager.
        seeds = getattr(tsdf, "seeds", None) if tsdf is not None else None
        if seeds:
            seed_ids = [str(getattr(s, "seed_id", s) if not isinstance(s, str) else s)
                        for s in list(seeds)[:20]]
            lines.append("seeds: " + ", ".join(seed_ids))
        elif tsdf is not None:
            rooms = getattr(tsdf, "room_regions", None) or []
            if rooms:
                room_ids = [
                    str(getattr(room, "room_id"))
                    for room in list(rooms)[:20]
                    if getattr(room, "room_id", None) is not None
                ]
                if room_ids:
                    lines.append("seeds: " + ", ".join(room_ids))
            else:
                lines.append("seeds: none")

        # Nearby objects — from scene.objects, limited to class names.
        if scene is not None:
            objects = _objs or {}
            if objects:
                names = []
                for obj in list(objects.values())[:30]:
                    if isinstance(obj, dict) and "class_name" in obj:
                        names.append(str(obj["class_name"]))
                if names:
                    lines.append("objects: " + ", ".join(sorted(set(names))))
                else:
                    lines.append("objects: none")
            else:
                lines.append("objects: none")

        return "\n".join(lines)
