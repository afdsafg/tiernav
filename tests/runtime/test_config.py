"""Tests for runtime provider/path configuration."""
from __future__ import annotations

import json

from pydantic import ValidationError

from src.tiernav_runtime.config import ProviderConfig, RuntimeConfig
from src.tiernav_runtime.contracts import RunSpec, RuntimeMode


def _minimal_spec(**overrides) -> RunSpec:
    base = dict(
        run_id="run-1",
        task_name="aeqa",
        dataset_split="dev",
        output_dir="/tmp/tiernav",
        planner_provider="mimo",
        planner_model="qwen3-vl-flash",
    )
    base.update(overrides)
    return RunSpec(**base)


def test_run_spec_carries_provider_injection_fields_and_runtime_mode():
    spec = _minimal_spec(
        runtime_mode=RuntimeMode.LEGACY,
        planner_base_url="https://api.example.com/v1",
        planner_api_key_env="QWEN_PLANNER_API_KEY",
    )

    assert spec.runtime_mode is RuntimeMode.LEGACY
    assert spec.planner_provider == "mimo"
    assert spec.planner_model == "qwen3-vl-flash"
    assert spec.planner_base_url == "https://api.example.com/v1"
    # Field holds the env var NAME, not a secret value.
    assert spec.planner_api_key_env == "QWEN_PLANNER_API_KEY"


def test_run_spec_defaults_to_graph_runtime_mode():
    spec = _minimal_spec()

    assert spec.runtime_mode is RuntimeMode.GRAPH
    assert spec.planner_base_url == ""
    assert spec.planner_api_key_env == ""


def test_run_spec_rejects_unknown_runtime_mode():
    try:
        _minimal_spec(runtime_mode="fancy")
    except ValidationError as exc:
        assert "runtime_mode" in str(exc)
    else:
        raise AssertionError("RunSpec accepted an unknown runtime_mode")


def test_run_spec_serializes_provider_fields_without_secrets():
    spec = _minimal_spec(
        planner_api_key_env="QWEN_PLANNER_API_KEY",
        planner_base_url="https://api.example.com/v1",
    )

    payload = json.loads(spec.model_dump_json())

    assert payload["planner_api_key_env"] == "QWEN_PLANNER_API_KEY"
    assert payload["planner_base_url"] == "https://api.example.com/v1"
    # No secret material leaks into the serialized form.
    assert all("sk-" not in str(v) for v in payload.values())


def test_runtime_config_keeps_provider_settings_injected():
    cfg = RuntimeConfig(
        provider=ProviderConfig(
            api_key_env="TEST_KEY",
            base_url_env="TEST_BASE_URL",
            model_env="TEST_MODEL",
        )
    )
    assert cfg.provider.api_key_env == "TEST_KEY"
    assert cfg.provider.base_url_env == "TEST_BASE_URL"
    assert cfg.provider.model_env == "TEST_MODEL"


def test_provider_config_keeps_env_var_reference_not_secret():
    provider = ProviderConfig(
        api_key_env="QWEN_PLANNER_API_KEY",
        base_url_env="QWEN_PLANNER_BASE_URL",
        model_env="QWEN_PLANNER_MODEL",
    )

    assert provider.api_key_env == "QWEN_PLANNER_API_KEY"
    # The config object itself never stores resolved values.
    assert not hasattr(provider, "api_key")
    dump = json.loads(provider.model_dump_json())
    assert "api_key_env" in dump
    assert "api_key" not in dump
    assert "base_url" not in dump
    assert "model" not in dump


def test_provider_config_resolve_api_key_reads_env(monkeypatch):
    provider = ProviderConfig(api_key_env="QWEN_PLANNER_API_KEY")

    monkeypatch.setenv("QWEN_PLANNER_API_KEY", "sk-test-secret")
    assert provider.resolve_api_key() == "sk-test-secret"

    monkeypatch.delenv("QWEN_PLANNER_API_KEY", raising=False)
    assert provider.resolve_api_key() == ""


def test_provider_config_resolve_api_key_uses_injected_environ():
    provider = ProviderConfig(api_key_env="OPENAI_API_KEY")

    assert provider.resolve_api_key({"OPENAI_API_KEY": "sk-injected"}) == "sk-injected"
    assert provider.resolve_api_key({}) == ""


def test_provider_config_resolve_base_url_reads_env(monkeypatch):
    provider = ProviderConfig(base_url_env="QWEN_PLANNER_BASE_URL")

    monkeypatch.setenv("QWEN_PLANNER_BASE_URL", "https://api.example.com/v1")
    assert provider.resolve_base_url() == "https://api.example.com/v1"

    monkeypatch.delenv("QWEN_PLANNER_BASE_URL", raising=False)
    assert provider.resolve_base_url() == ""


def test_provider_config_resolve_model_reads_env(monkeypatch):
    provider = ProviderConfig(model_env="QWEN_PLANNER_MODEL")

    monkeypatch.setenv("QWEN_PLANNER_MODEL", "qwen3-vl-flash")
    assert provider.resolve_model() == "qwen3-vl-flash"

    monkeypatch.delenv("QWEN_PLANNER_MODEL", raising=False)
    assert provider.resolve_model() == ""


def test_provider_config_with_empty_env_vars_returns_empty():
    provider = ProviderConfig()
    assert provider.api_key_env == ""
    assert provider.base_url_env == ""
    assert provider.model_env == ""
    assert provider.resolve_api_key() == ""
    assert provider.resolve_base_url() == ""
    assert provider.resolve_model() == ""


def test_provider_config_rejects_unknown_fields():
    try:
        ProviderConfig(provider="mimo", model="qwen3-vl-flash")
    except ValidationError as exc:
        assert "provider" in str(exc)
    else:
        raise AssertionError("ProviderConfig accepted removed provider/model fields")


def test_runtime_config_requires_provider():
    try:
        RuntimeConfig()
    except ValidationError as exc:
        assert "provider" in str(exc)
    else:
        raise AssertionError("RuntimeConfig accepted a missing provider")
