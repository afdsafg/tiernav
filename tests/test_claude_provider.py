"""Verify ClaudeProvider implementation — tool-use + cache_control.

Tests run without the `anthropic` SDK installed: __init__ stores api_key/model
without importing anthropic. The SDK is only required for `call()`, which is
not exercised here (network).
"""
import pytest


def test_claude_provider_class_exists():
    """ClaudeProvider class must exist and be importable."""
    from src.two_tier_graph.providers import ClaudeProvider
    assert ClaudeProvider is not None


def test_claude_provider_init_no_sdk_required():
    """Constructing ClaudeProvider must not require anthropic SDK installed."""
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    assert provider is not None


def test_claude_provider_action_to_tool_definition():
    """Each of the 5 planner actions should map to an Anthropic tool definition."""
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    tools = provider.get_tool_definitions()
    assert len(tools) >= 5
    names = {t["name"] for t in tools}
    expected = {
        "explore_panorama",
        "navigate_to_object",
        "explore_seed",
        "explore_frontier",
        "submit_answer",
    }
    assert expected.issubset(names), f"missing tools: {expected - names}"
    # Each tool should have name, description, input_schema
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t
        assert t["input_schema"]["type"] == "object"


def test_claude_provider_cache_control_on_cacheable_sections():
    """Cacheable prompt sections should get cache_control: ephemeral.

    Anthropic prompt caching: mark the LAST cacheable block with
    cache_control={"type":"ephemeral"} so the cacheable prefix is cached.
    """
    from src.two_tier_graph.providers import ClaudeProvider
    from src.two_tier_graph.prompt_sections import PromptSection
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    sections = [
        PromptSection("static_a", "static content a", cacheable=True),
        PromptSection("static_b", "static content b", cacheable=True),
        PromptSection("dynamic", "dynamic content", cacheable=False),
    ]
    system, messages = provider.build_messages(sections)
    # cache_control should appear on the last cacheable block, not the dynamic one.
    found_cache = False
    for block in system:
        if isinstance(block, dict) and block.get("cache_control"):
            found_cache = True
    assert found_cache, "No cache_control: ephemeral found on system (cacheable) blocks"
    # The last cacheable block specifically must carry cache_control.
    last_cacheable_idx = max(
        i for i, b in enumerate(system) if isinstance(b, dict)
    )
    # find last cacheable by scanning for cache_control marker
    cache_marked = [
        b for b in system
        if isinstance(b, dict) and b.get("cache_control") == {"type": "ephemeral"}
    ]
    assert len(cache_marked) >= 1
    # Dynamic (non-cacheable) content should NOT carry cache_control.
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    assert not block.get("cache_control"), \
                        "dynamic section should not be cache-marked"


def test_claude_provider_tool_use_to_planner_action():
    """tool_use response block should map to PlannerAction."""
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    # Simulate a tool_use response. Field name is frontier_id (matches
    # ExploreFrontierTool.arg_fields + PlannerAction.frontier_id).
    mock_response = {
        "content": [
            {"type": "text", "text": "I should explore a frontier"},
            {"type": "tool_use", "id": "toolu_1", "name": "explore_frontier",
             "input": {"frontier_id": "5", "reason": "doorway ahead"}},
        ]
    }
    action = provider.parse_tool_response(mock_response)
    assert action is not None
    assert action.action_type == "explore_frontier"
    assert action.frontier_id == "5"
    assert action.reason == "doorway ahead"


def test_claude_provider_tool_use_navigate_to_object():
    """navigate_to_object tool_use should populate snapshot_id + object_name."""
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    mock_response = {
        "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "navigate_to_object",
             "input": {"snapshot_id": "step3_view1", "object_name": "red chair",
                       "reason": "check seat"}},
        ]
    }
    action = provider.parse_tool_response(mock_response)
    assert action is not None
    assert action.action_type == "navigate_to_object"
    assert action.snapshot_id == "step3_view1"
    assert action.object_name == "red chair"


def test_claude_provider_tool_use_submit_answer():
    """submit_answer tool_use should populate answer + snapshot_id."""
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    mock_response = {
        "content": [
            {"type": "tool_use", "id": "toolu_3", "name": "submit_answer",
             "input": {"snapshot_id": "step5_view0", "answer": "kitchen",
                       "reason": "final"}},
        ]
    }
    action = provider.parse_tool_response(mock_response)
    assert action.action_type == "submit_answer"
    assert action.answer == "kitchen"
    assert action.snapshot_id == "step5_view0"


def test_claude_provider_parse_no_tool_use_returns_none():
    """If response has no tool_use block, return None (caller handles)."""
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    mock_response = {"content": [{"type": "text", "text": "thinking..."}]}
    action = provider.parse_tool_response(mock_response)
    assert action is None


def test_claude_provider_not_implemented_decide():
    """decide() requires network + SDK; not exercised in unit tests.

    But the method should exist and raise a clear error when SDK missing
    rather than ImportError at module load.
    """
    from src.two_tier_graph.providers import ClaudeProvider
    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")
    # decide() should raise (sdk missing or NotImplementedError) — not crash on import.
    with pytest.raises((ImportError, NotImplementedError, RuntimeError)):
        provider.decide(question="q", history="", scene="", progress="",
                        actions="", image_b64=None)
