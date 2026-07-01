"""Planner adapter bridging legacy PlannerAction to runtime contracts."""
from __future__ import annotations

from typing import Any, Optional

from .config import ProviderConfig
from .contracts import PlannerDecision, PlannerMessage

# Fields collected into PlannerDecision.arguments when non-None on the
# source action. Kept explicit so the adapter is stable across PlannerAction
# refactors.
_ARGUMENT_FIELDS = (
    "snapshot_id",
    "object_name",
    "seed_id",
    "frontier_id",
    "view_idx",
    "answer",
)


def planner_action_to_decision(action: Any) -> PlannerDecision:
    """Convert a legacy PlannerAction into a validated PlannerDecision.

    Drops None optional fields and lets PlannerDecision clamp confidence.
    """
    confidence = getattr(action, "confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    arguments = {
        field: getattr(action, field, None)
        for field in _ARGUMENT_FIELDS
        if getattr(action, field, None) is not None
    }

    return PlannerDecision(
        action_type=getattr(action, "action_type", ""),
        reasoning=getattr(action, "reason", "") or "",
        expected=getattr(action, "expected", "") or "",
        confidence=confidence,
        arguments=arguments,
    )


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from VLM output."""
    s = text.strip()
    if s.startswith("```"):
        # Drop opening fence line (```json or ```)
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        # Drop closing fence
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s


def _call_vlm(messages: list[dict], **kwargs: Any) -> str:
    """Indirection so tests can monkeypatch the OpenAI-compatible transport.

    Delegates to ``src.agent_workflow.call_vlm``. Kept as a module-level
    function (not an import) so the planner module has a single seam to
    patch at the call boundary.
    """
    from src.agent_workflow import call_vlm

    return call_vlm(messages, **kwargs)


def _coerce_messages(prompt_or_messages: Any) -> list[dict]:
    """Return OpenAI-compatible messages from text, dicts, or PlannerMessage models."""
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]

    if isinstance(prompt_or_messages, PlannerMessage):
        return [prompt_or_messages.model_dump(mode="json", exclude_none=True)]

    if isinstance(prompt_or_messages, list):
        messages: list[dict] = []
        for item in prompt_or_messages:
            if isinstance(item, PlannerMessage):
                messages.append(item.model_dump(mode="json", exclude_none=True))
            elif isinstance(item, dict):
                # Preserve existing transport behavior: copy the message envelope
                # while leaving nested multimodal content untouched.
                messages.append(dict(item))
            else:
                raise TypeError(
                    "planner messages must be dict or PlannerMessage, got "
                    + type(item).__name__
                )
        return messages

    raise TypeError(
        "prompt must be str, PlannerMessage, or list of messages, got "
        + type(prompt_or_messages).__name__
    )


class PlannerClient:
    """Provider-agnostic planner transport.

    Reads ``api_key`` / ``base_url`` / ``model`` from the injected
    ``ProviderConfig`` lazily at call time, so environment changes are
    picked up between calls. Explicit constructor overrides win over
    config resolution. Calls the existing OpenAI-compatible transport
    (``call_vlm`` in ``src.agent_workflow``) — no hard-coded vendor
    endpoints.
    """

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

    def resolve_api_key(self) -> str:
        return self._api_key if self._api_key is not None else self.provider.resolve_api_key()

    def resolve_base_url(self) -> str:
        return self._base_url if self._base_url is not None else self.provider.resolve_base_url()

    def resolve_model(self) -> str:
        return self._model if self._model is not None else self.provider.resolve_model()

    def call_vlm(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Call the OpenAI-compatible transport with resolved settings."""
        return _call_vlm(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.resolve_api_key(),
            base_url=self.resolve_base_url(),
            model_name=self.resolve_model(),
        )

    def decide(self, prompt: Any, *, retries: int = 0) -> PlannerDecision:
        """Call the VLM with the compiled prompt and return a PlannerDecision.

        Accepts a legacy string prompt, a ``PlannerMessage``, or a list of
        OpenAI-compatible messages, calls the transport, parses the JSON
        response, and maps it through ``planner_action_to_decision`` for
        legacy PlannerAction compatibility.

        On JSON parse errors or missing ``action_type``, retries up to
        ``retries`` times before returning a terminal ``submit_answer``
        with confidence 0.0 to avoid infinite loops on unparseable planner
        output. ``retries=0`` (default) preserves the original fail-fast
        behavior.
        """
        import json as _json

        messages = _coerce_messages(prompt)
        last_error_decision: PlannerDecision | None = None

        for attempt in range(retries + 1):
            try:
                raw = self.call_vlm(messages)
            except Exception:
                last_error_decision = PlannerDecision(
                    action_type="submit_answer",
                    reasoning="planner_call_failed",
                    confidence=0.0,
                    arguments={"failure_reason": "planner_call_failed",
                               "attempt": attempt + 1},
                )
                continue

            try:
                cleaned = _strip_code_fences(raw)
                parsed = _json.loads(cleaned.strip())
            except _json.JSONDecodeError:
                last_error_decision = PlannerDecision(
                    action_type="submit_answer",
                    reasoning="planner_parse_error",
                    confidence=0.0,
                    arguments={"failure_reason": "planner_parse_error",
                               "raw": raw[:500], "attempt": attempt + 1},
                )
                continue

            if not isinstance(parsed, dict) or not parsed.get("action_type"):
                reason = ("planner_response_not_dict" if not isinstance(parsed, dict)
                          else "planner_missing_action_type")
                last_error_decision = PlannerDecision(
                    action_type="submit_answer",
                    reasoning=reason,
                    confidence=0.0,
                    arguments={"failure_reason": reason, "attempt": attempt + 1},
                )
                continue

            return planner_action_to_decision(
                type("PlannerAction", (), parsed)()
            )

        return last_error_decision  # type: ignore[return-value]
