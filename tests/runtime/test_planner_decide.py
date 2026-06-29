"""Tests for PlannerClient.decide() — production VLM planner path."""
import pytest
from unittest.mock import patch

from src.tiernav_runtime.config import ProviderConfig
from src.tiernav_runtime.contracts import PlannerDecision
from src.tiernav_runtime.planner import PlannerClient


class TestPlannerClientDecide:
    def test_decide_calls_vlm_and_returns_decision(self):
        cfg = ProviderConfig(
            api_key_env="TEST_KEY",
            base_url_env="TEST_BASE_URL",
            model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")

        fake_response = (
            '{"action_type": "explore_panorama", '
            '"reason": "Need to observe surroundings", '
            '"expected": "Get room layout", '
            '"object_name": "chair"}'
        )

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value=fake_response,
        ) as mock_vlm:
            decision = client.decide("Test prompt")

        mock_vlm.assert_called_once()
        call_args = mock_vlm.call_args[0][0]
        assert any("Test prompt" in msg.get("content", "") for msg in call_args)

        assert isinstance(decision, PlannerDecision)
        assert decision.action_type == "explore_panorama"
        assert decision.reasoning == "Need to observe surroundings"
        assert decision.arguments.get("object_name") == "chair"

    def test_decide_handles_invalid_json(self):
        cfg = ProviderConfig(
            api_key_env="TEST_KEY", base_url_env="TEST_BASE_URL", model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value="not valid json {{{",
        ):
            decision = client.decide("Test prompt")

        assert decision.action_type == "submit_answer"
        assert decision.confidence == 0.0
        assert "planner_parse_error" in decision.arguments.get("failure_reason", "")

    def test_decide_handles_missing_action_type(self):
        cfg = ProviderConfig(
            api_key_env="TEST_KEY", base_url_env="TEST_BASE_URL", model_env="TEST_MODEL",
        )
        client = PlannerClient(cfg, api_key="sk-test", base_url="http://test", model="test-model")

        with patch(
            "src.tiernav_runtime.planner._call_vlm",
            return_value='{"reason": "no action"}',
        ):
            decision = client.decide("Test prompt")

        assert decision.action_type == "submit_answer"
        assert decision.confidence == 0.0
