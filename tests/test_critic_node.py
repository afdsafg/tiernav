"""Verify critic node — veto + re-decision with feedback."""
import pytest


def test_critic_node_exists():
    """critic_node function exists."""
    from src.two_tier_graph.nodes import critic_node
    assert callable(critic_node)


def test_critic_node_passthrough_when_disabled():
    """When critic.enabled=false, critic_node should passthrough (no veto)."""
    from src.two_tier_graph.nodes import critic_node
    # Minimal state + config mock
    state = {
        "current_action": {"action_type": "explore_frontier"},
        "rounds_used": 1,
    }
    config = {"configurable": {"resources": type("R", (), {"critic_enabled": False})()}}
    result = critic_node(state, config)
    # Should not veto — return empty or passthrough
    assert result is not None
    assert not result.get("critic_veto", False)


def test_critic_node_can_veto():
    """When critic.enabled=true, critic_node can veto with feedback."""
    from src.two_tier_graph.nodes import critic_node
    state = {
        "current_action": {"action_type": "explore_frontier"},
        "rounds_used": 1,
    }
    config = {"configurable": {"resources": type("R", (), {"critic_enabled": True})()}}
    result = critic_node(state, config)
    # Result should have critic_veto field (True or False)
    assert "critic_veto" in result or result == {}


def test_after_critic_edge_exists():
    """after_critic edge function exists."""
    from src.two_tier_graph.edges import after_critic
    assert callable(after_critic)


def test_after_critic_routes_on_veto():
    """after_critic should route back to planner on veto."""
    from src.two_tier_graph.edges import after_critic
    state = {"critic_veto": True}
    assert after_critic(state) == "planner"


def test_after_critic_routes_forward_no_veto():
    """after_critic should route to executor when no veto."""
    from src.two_tier_graph.edges import after_critic
    state = {"critic_veto": False}
    assert after_critic(state) == "loop_guard"
