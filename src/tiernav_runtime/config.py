"""Runtime configuration for provider injection.

These models are runtime construction inputs, not serialized episode
contracts, so they are intentionally kept out of ``PUBLIC_MODELS`` and the
JSON schema dump. They never store secret values: API keys, base URLs, and
model names are resolved at call time from the environment variables named
by ``api_key_env`` / ``base_url_env`` / ``model_env``.
"""
from __future__ import annotations

import os
from typing import Optional

from .contracts import RuntimeModel


class ProviderConfig(RuntimeModel):
    """Connection spec for an OpenAI-compatible planner/VLM provider.

    Each field holds the NAME of an environment variable, never the resolved
    value, so a serialized config never carries secrets or hard-coded
    endpoints/models. ``resolve_*`` read lazily at call time.
    """

    api_key_env: str = ""
    base_url_env: str = ""
    model_env: str = ""

    def resolve_api_key(self, environ: Optional[dict[str, str]] = None) -> str:
        """Return the API key from the configured env var, or empty string."""
        env = os.environ if environ is None else environ
        return env.get(self.api_key_env, "") if self.api_key_env else ""

    def resolve_base_url(self, environ: Optional[dict[str, str]] = None) -> str:
        """Return the base URL from the configured env var, or empty string."""
        env = os.environ if environ is None else environ
        return env.get(self.base_url_env, "") if self.base_url_env else ""

    def resolve_model(self, environ: Optional[dict[str, str]] = None) -> str:
        """Return the model name from the configured env var, or empty string."""
        env = os.environ if environ is None else environ
        return env.get(self.model_env, "") if self.model_env else ""


class RuntimeConfig(RuntimeModel):
    """Runtime-wide construction config.

    ``provider`` carries the injected planner connection spec; the runtime
    resolves its values lazily from the environment.
    """

    provider: ProviderConfig
