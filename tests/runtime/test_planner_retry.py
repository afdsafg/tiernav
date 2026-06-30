"""PlannerClient.decide retries on parse errors up to planner_retries."""
from unittest.mock import patch
from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.planner import PlannerClient


def test_decide_retries_on_parse_error():
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    calls = {"n": 0}
    def fake_vlm(messages, **kw):
        calls["n"] += 1
        return "not json {{{" if calls["n"] == 1 else '{"action_type": "explore_panorama", "reason": "ok"}'

    with patch("src.tiernav_runtime.planner._call_vlm", side_effect=fake_vlm):
        decision = client.decide("p", retries=1)

    assert calls["n"] == 2
    assert decision.action_type == "explore_panorama"


def test_decide_falls_back_after_retry_budget_exhausted():
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    with patch("src.tiernav_runtime.planner._call_vlm", return_value="not json {{{"):
        decision = client.decide("p", retries=2)

    assert decision.action_type == "submit_answer"
    assert "planner_parse_error" in decision.arguments.get("failure_reason", "")


def test_decide_no_retry_by_default():
    """Default retries=0: single call, immediate fallback on parse error."""
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    with patch("src.tiernav_runtime.planner._call_vlm", return_value="not json {{{") as mock:
        decision = client.decide("p")

    assert mock.call_count == 1
    assert decision.action_type == "submit_answer"


def test_decide_retries_on_call_exception():
    """Retry when call_vlm raises, recover on the next attempt."""
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    calls = {"n": 0}
    def fake_vlm(messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient VLM outage")
        return '{"action_type": "explore_panorama", "reason": "ok"}'

    with patch("src.tiernav_runtime.planner._call_vlm", side_effect=fake_vlm):
        decision = client.decide("p", retries=1)

    assert calls["n"] == 2
    assert decision.action_type == "explore_panorama"


def test_decide_falls_back_when_all_calls_raise():
    """All attempts raise → fallback submit_answer with planner_call_failed."""
    cfg = ProviderConfig(api_key_env="T", base_url_env="T", model_env="T")
    client = PlannerClient(cfg, api_key="k", base_url="http://t", model="m")

    with patch("src.tiernav_runtime.planner._call_vlm",
               side_effect=RuntimeError("down")):
        decision = client.decide("p", retries=2)

    assert decision.action_type == "submit_answer"
    assert "planner_call_failed" in decision.arguments.get("failure_reason", "")
