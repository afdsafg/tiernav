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
    """Future provider using `anthropic` SDK with native tool-use.

    NOT implemented this phase (out of scope per plan). The 5 actions become
    tool definitions; `decide()` issues a tool-calling request and maps the
    returned `tool_use` block to `PlannerAction` (no JSON-in-text parsing).
    `decide_raw()` sends a plain text response request.

    Implemented in a LATER isolated experiment behind `cfg.llm.provider`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ClaudeProvider is not implemented in this phase. Use MimoProvider. "
            "See plan §5 (LLM provider interface) and §8 (out of scope #4)."
        )

    def decide(self, *args, **kwargs):
        raise NotImplementedError

    def decide_raw(self, *args, **kwargs):
        raise NotImplementedError


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
        raise NotImplementedError(
            "Claude provider is not implemented in this phase. Set cfg.llm.provider='mimo' "
            "or omit cfg.llm to use the default mimo provider."
        )
    if provider_name != "mimo":
        raise ValueError(f"Unknown llm provider: {provider_name!r} (expected 'mimo' or 'claude')")
    return MimoProvider(planner)
