"""Verify transition.reason is first-class state field."""
from src.two_tier_graph.state import TransitionReason, Transition


def test_transition_reason_enum_has_all_values():
    assert TransitionReason.CONTINUE == "continue"
    assert TransitionReason.ROUND_BUDGET == "round_budget"
    assert TransitionReason.EXHAUSTED == "exhausted"
    assert TransitionReason.STEP_BUDGET == "step_budget"
    # P3 will activate these (defined now, unused until P3)
    assert TransitionReason.STALL_RECOVERY == "stall_recovery"
    assert TransitionReason.VERIFY_BEFORE_FALLBACK == "verify_before_fallback"


def test_transition_dataclass():
    t = Transition(reason=TransitionReason.CONTINUE, from_node="memory_update",
                   to_node="build_context", round_idx=2)
    assert t.reason == TransitionReason.CONTINUE
    assert t.round_idx == 2


def test_state_has_transition_fields():
    from src.two_tier_graph.state import TwoTierState
    assert "last_transition" in TwoTierState.__annotations__
    assert "transition_log" in TwoTierState.__annotations__


def test_after_memory_reads_last_transition():
    """after_memory should route based on last_transition.reason, not recompute."""
    from src.two_tier_graph.edges import after_memory
    # Round budget
    state = {"last_transition": {"reason": "round_budget"}}
    assert after_memory(state) == "fallback_submit"
    # Exhausted
    state = {"last_transition": {"reason": "exhausted"}}
    assert after_memory(state) == "continue"
    # Step budget
    state = {"last_transition": {"reason": "step_budget"}}
    assert after_memory(state) == "fallback_submit"
    # Continue
    state = {"last_transition": {"reason": "continue"}}
    assert after_memory(state) == "continue"
