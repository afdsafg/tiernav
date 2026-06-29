"""Tests for the provider-injected PlannerClient."""
from __future__ import annotations

import os

import pytest

from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.planner import PlannerClient


def test_planner_client_uses_injected_provider_settings():
    cfg = ProviderConfig(
        api_key_env="PLAN_API_KEY",
        base_url_env="PLAN_BASE_URL",
        model_env="PLAN_MODEL",
    )
    client = PlannerClient(provider=cfg)
    assert client.provider.api_key_env == "PLAN_API_KEY"
    assert client.provider.base_url_env == "PLAN_BASE_URL"
    assert client.provider.model_env == "PLAN_MODEL"


def test_explicit_overrides_win_over_provider_config(monkeypatch):
    cfg = ProviderConfig(
        api_key_env="PLAN_API_KEY",
        base_url_env="PLAN_BASE_URL",
        model_env="PLAN_MODEL",
    )
    monkeypatch.setenv("PLAN_API_KEY", "sk-from-env")
    monkeypatch.setenv("PLAN_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setenv("PLAN_MODEL", "env-model")

    client = PlannerClient(
        provider=cfg,
        api_key="sk-explicit",
        base_url="https://explicit.example.com/v1",
        model="explicit-model",
    )
    assert client.resolve_api_key() == "sk-explicit"
    assert client.resolve_base_url() == "https://explicit.example.com/v1"
    assert client.resolve_model() == "explicit-model"


def test_resolution_is_lazy_and_picks_up_env_changes(monkeypatch):
    cfg = ProviderConfig(
        api_key_env="PLAN_API_KEY",
        base_url_env="PLAN_BASE_URL",
        model_env="PLAN_MODEL",
    )
    client = PlannerClient(provider=cfg)

    monkeypatch.setenv("PLAN_API_KEY", "sk-first")
    assert client.resolve_api_key() == "sk-first"

    monkeypatch.setenv("PLAN_API_KEY", "sk-second")
    assert client.resolve_api_key() == "sk-second"


def test_call_drives_openai_compatible_transport(monkeypatch):
    """PlannerClient.call_vlm delegates to src.agent_workflow.call_vlm
    with resolved provider settings; no hard-coded vendor endpoint."""
    captured: dict = {}

    def fake_call_vlm(messages, **kwargs):
        captured["messages"] = messages
        captured["api_key"] = kwargs.get("api_key")
        captured["base_url"] = kwargs.get("base_url")
        captured["model_name"] = kwargs.get("model_name")
        return '{"action":"explore_panorama","reasoning":"ok","confidence":0.5}'

    cfg = ProviderConfig(
        api_key_env="PLAN_API_KEY",
        base_url_env="PLAN_BASE_URL",
        model_env="PLAN_MODEL",
    )
    monkeypatch.setenv("PLAN_API_KEY", "sk-transport")
    monkeypatch.setenv("PLAN_BASE_URL", "https://transport.example.com/v1")
    monkeypatch.setenv("PLAN_MODEL", "transport-model")

    import src.tiernav_runtime.planner as planner_mod
    monkeypatch.setattr(planner_mod, "_call_vlm", fake_call_vlm)

    client = PlannerClient(provider=cfg)
    messages = [{"role": "user", "content": "hi"}]
    response = client.call_vlm(messages)

    assert captured["api_key"] == "sk-transport"
    assert captured["base_url"] == "https://transport.example.com/v1"
    assert captured["model_name"] == "transport-model"
    assert captured["messages"] is messages
    assert "explore_panorama" in response
