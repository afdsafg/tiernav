"""LLMProvider — abstraction over the Planner LLM.

Two methods because the codebase has two call shapes:
  - `decide()` → returns a parsed `PlannerAction` (main planner decision).
  - `decide_raw()` → returns raw text (Stage 6.5 frontier selection at
    agent_workflow.py:1338, submit_best_guess fallback at :1681) which have
    their own tolerant parsers.

`MimoProvider` is the default this phase — a thin adapter delegating to the
existing `Planner` class (agent_planner.py:68). Behavior is byte-identical to
the legacy `planner.decide(...)` and `call_vlm(...)` calls.

`ClaudeProvider` is specified but NOT implemented (out of scope per plan §5).
The interface exists so `planner_node` is provider-agnostic; swapping is a
config change, not a graph change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Planner LLM abstraction. Returns a parsed PlannerAction or raw text."""

    @abstractmethod
    def decide(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
        image_b64s: Optional[list[str]] = None,
        image_b64: Optional[str] = None,
    ):
        """Call LLM with 4-component prompt + images, return parsed PlannerAction.

        Mirrors `Planner.decide` signature (agent_planner.py:81) so MimoProvider
        can delegate directly.
        """
        ...

    @abstractmethod
    def decide_raw(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Raw text response for non-action VLM calls (Stage 6.5 frontier,
        submit_best_guess fallback). Returns unparsed text so the caller can
        apply its own tolerant parser (e.g. _parse_stage65_frontier_response)."""
        ...


class MimoProvider(LLMProvider):
    """Default provider — delegates to the existing Planner + call_vlm.

    Byte-identical behavior to the legacy `planner.decide(...)` and
    `call_vlm(messages, image_b64=..., ...)` calls.
    """

    def __init__(self, planner):
        # `planner` is a src.agent_planner.Planner instance.
        self._planner = planner

    def decide(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
        image_b64s: Optional[list[str]] = None,
        image_b64: Optional[str] = None,
    ):
        # Delegate verbatim to Planner.decide (agent_planner.py:81-119).
        # Planner.decide accepts both `image_b64` and `image_b64s` and
        # composes the messages itself; we forward both for fidelity.
        return self._planner.decide(
            question=question,
            history=history,
            scene=scene,
            progress=progress,
            actions=actions,
            image_b64=image_b64,
            image_b64s=image_b64s,
        )

    def decide_raw(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        # Delegate to the module-level call_vlm (agent_workflow.py:179).
        from src.agent_workflow import call_vlm
        return call_vlm(
            messages,
            image_b64=image_b64,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class ClaudeProvider(LLMProvider):
    """Provider using `anthropic` SDK with native tool-use.

    The 5 planner actions become Anthropic tool definitions; `decide()` issues
    a tool-calling request and maps the returned `tool_use` block to
    `PlannerAction` (no JSON-in-text parsing). `decide_raw()` sends a plain
    text response request. `cache_control: {"type": "ephemeral"}` is attached
    to the last cacheable prompt section so the static prefix is cached.

    `anthropic` is imported lazily inside `_client()` / `decide()` / `decide_raw()`
    so the module loads without the SDK installed — unit tests for tool
    definition building, cache_control marking, and tool_use parsing run
    without network or SDK.
    """

    # 5 planner actions → Anthropic tool input_schema. arg_fields mirror
    # src.two_tier_graph.tools (ExplorePanoramaTool … SubmitAnswerTool).
    # `reason` is accepted on every tool so the planner can justify; it maps
    # to PlannerAction.reason.
    _TOOL_SCHEMAS = [
        {
            "name": "explore_panorama",
            "description": "Re-orient with a full 8-view panorama of the current position.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "why re-orient now"},
                },
                "required": [],
            },
        },
        {
            "name": "navigate_to_object",
            "description": "Navigate toward an object visible in an attached current-view snapshot.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "snapshot_id": {"type": "string", "description": "e.g. step12_view1"},
                    "object_name": {"type": "string", "description": "target visible object"},
                    "reason": {"type": "string"},
                },
                "required": ["snapshot_id", "object_name"],
            },
        },
        {
            "name": "explore_seed",
            "description": "Navigate to a seed viewpoint.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "seed_id": {"type": "string", "description": "seed viewpoint id"},
                    "reason": {"type": "string"},
                },
                "required": ["seed_id"],
            },
        },
        {
            "name": "explore_frontier",
            "description": "Navigate to one of the current unexplored frontiers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "frontier_id": {"type": "string", "description": "frontier id"},
                    "reason": {"type": "string"},
                },
                "required": ["frontier_id"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer (terminal).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "snapshot_id": {"type": "string", "description": "e.g. stepN_viewM"},
                    "answer": {"type": "string", "description": "final answer"},
                    "reason": {"type": "string"},
                },
                "required": ["snapshot_id", "answer"],
            },
        },
    ]

    def __init__(self, api_key: str, model: str, **kwargs):
        # Do NOT import anthropic here — module must load without the SDK.
        self._api_key = api_key
        self._model = model
        self._max_tokens = kwargs.get("max_tokens", 4096)
        self._temperature = kwargs.get("temperature", 0.3)
        self._client = None  # lazy

    # ── SDK bootstrap ────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-import anthropic and build client. Raises ImportError if missing."""
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "anthropic SDK not installed. Install with `pip install anthropic` "
                    "or use MimoProvider (cfg.llm.provider='mimo')."
                ) from e
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ── Tool definitions ─────────────────────────────────────────────────

    def get_tool_definitions(self) -> list[dict]:
        """Return the 5 action tools as Anthropic tool definitions."""
        # Return fresh dicts so callers can't mutate our class-level schemas.
        import copy
        return copy.deepcopy(self._TOOL_SCHEMAS)

    # ── Messages + cache_control ─────────────────────────────────────────

    def build_messages(self, sections) -> tuple[list[dict], list[dict]]:
        """Build Anthropic (system, messages) from PromptSections.

        Cacheable sections go into the system prompt (the cacheable prefix);
        non-cacheable sections become the user message. The LAST cacheable
        system block is marked `cache_control: {"type": "ephemeral"}` so
        Anthropic caches the static prefix across rounds.

        Returns (system_blocks, messages) where system_blocks is a list of
        {"type":"text","text":...} dicts and messages is a list of role/content
        dicts ready for client.messages.create.
        """
        from src.two_tier_graph.prompt_sections import PromptSection

        system_blocks: list[dict] = []
        dynamic_texts: list[str] = []
        for s in sections:
            # PromptSection.content may be str or list[str] (e.g. image_b64 list).
            text = s.content if isinstance(s.content, str) else "\n".join(s.content)
            if s.cacheable:
                system_blocks.append({"type": "text", "text": text})
            else:
                dynamic_texts.append(text)

        # Mark the last cacheable block with cache_control: ephemeral.
        if system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        messages: list[dict] = []
        if dynamic_texts:
            messages.append({"role": "user", "content": "\n\n".join(dynamic_texts)})
        return system_blocks, messages

    # ── tool_use → PlannerAction ─────────────────────────────────────────

    def parse_tool_response(self, response) -> Optional["PlannerAction"]:
        """Extract the first `tool_use` block and map to PlannerAction.

        Returns None if no tool_use block is present (caller must handle,
        e.g. fall back to decide_raw-style text parsing).
        """
        from src.agent_planner import PlannerAction

        content = response.get("content") if isinstance(response, dict) else None
        if not content or not isinstance(content, list):
            return None
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = block.get("input", {}) or {}
            return PlannerAction(
                action_type=name,
                reason=inp.get("reason", ""),
                confidence=float(inp.get("confidence", 0.0)) if inp.get("confidence") is not None else 0.0,
                snapshot_id=inp.get("snapshot_id"),
                object_name=inp.get("object_name"),
                seed_id=inp.get("seed_id"),
                frontier_id=inp.get("frontier_id"),
                view_idx=inp.get("view_idx"),
                answer=inp.get("answer"),
                expected=inp.get("expected"),
            )
        return None

    # ── LLMProvider interface ────────────────────────────────────────────

    def decide(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
        image_b64s: Optional[list[str]] = None,
        image_b64: Optional[str] = None,
    ):
        """Call Claude with tool-use enabled; map tool_use → PlannerAction.

        Falls back to None if the model returns no tool_use block; callers
        that need text should use decide_raw().
        """
        from src.two_tier_graph.prompt_sections import PromptSection

        # Build sections: cacheable prefix (task+actions+query) then dynamic.
        sections = [
            PromptSection("task_instruction", actions, cacheable=True),  # actions carries schema
            PromptSection("active_query", question, cacheable=True),
            PromptSection("history", history, cacheable=False),
            PromptSection("scene", scene, cacheable=False),
            PromptSection("progress", progress, cacheable=False),
        ]
        system, messages = self.build_messages(sections)

        # Attach images (if any) to the user message as image content blocks.
        if image_b64s or image_b64:
            imgs = list(image_b64s or [])
            if image_b64:
                imgs.append(image_b64)
            user_content: list = []
            for b64 in imgs:
                user_content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
            # append existing dynamic text
            if messages and messages[0]["role"] == "user":
                user_content.insert(0, {"type": "text", "text": messages[0]["content"]})
            else:
                user_content.insert(0, {"type": "text", "text": ""})
            messages = [{"role": "user", "content": user_content}] + messages[1:]

        client = self._get_client()
        resp = client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            tools=self.get_tool_definitions(),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        # Normalize to a plain dict for parse_tool_response (works whether
        # the SDK returns a pydantic model or a dict).
        return self.parse_tool_response(_response_to_dict(resp))

    def decide_raw(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Plain text response (no tools). For tolerant-parser callers."""
        client = self._get_client()
        # Caller-supplied messages already in Anthropic shape; pass through.
        kwargs = dict(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if image_b64:
            kwargs["messages"] = [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": ""},
                ],
            }] + list(messages)
        resp = client.messages.create(**kwargs)
        # Extract text blocks.
        out = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                out.append(getattr(block, "text", ""))
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "".join(out)


def build_llm_provider(cfg, planner) -> LLMProvider:
    """Factory: pick provider by cfg.llm.provider. Defaults to mimo.

    `cfg` may be an OmegaConf/DictConfig or dict-like with attribute access.
    Falls back to mimo if `cfg.llm` or `cfg.llm.provider` is missing — this
    preserves the legacy behavior (no config change required to use the graph).
    """
    provider_name = "mimo"
    try:
        llm_cfg = getattr(cfg, "llm", None)
        if llm_cfg is None and isinstance(cfg, dict):
            llm_cfg = cfg.get("llm")
        if llm_cfg is not None:
            provider_name = getattr(llm_cfg, "provider", None) or (
                llm_cfg.get("provider") if isinstance(llm_cfg, dict) else None
            ) or "mimo"
    except Exception:
        pass

    if provider_name == "claude":
        # Read api_key + model from cfg.llm; fall back to env var for the key.
        import os
        api_key = (
            getattr(llm_cfg, "api_key", None)
            or (llm_cfg.get("api_key") if isinstance(llm_cfg, dict) else None)
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        model = (
            getattr(llm_cfg, "model", None)
            or (llm_cfg.get("model") if isinstance(llm_cfg, dict) else None)
            or "claude-sonnet-4-20250514"
        )
        return ClaudeProvider(api_key=api_key, model=model)
    if provider_name != "mimo":
        raise ValueError(f"Unknown llm provider: {provider_name!r} (expected 'mimo' or 'claude')")
    return MimoProvider(planner)


def _response_to_dict(resp) -> dict:
    """Normalize an Anthropic SDK response (pydantic model or dict) to a plain
    dict with a `content` list of block dicts, for parse_tool_response().
    """
    if isinstance(resp, dict):
        return resp
    content = []
    for block in getattr(resp, "content", []) or []:
        if isinstance(block, dict):
            content.append(block)
            continue
        # pydantic model: try .model_dump() then attribute fallback.
        dump = getattr(block, "model_dump", None)
        if callable(dump):
            content.append(dump())
            continue
        btype = getattr(block, "type", None)
        if btype == "text":
            content.append({"type": "text", "text": getattr(block, "text", "")})
        elif btype == "tool_use":
            content.append({
                "type": "tool_use",
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}) or {},
            })
    return {"content": content}
