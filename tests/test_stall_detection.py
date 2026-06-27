"""Verify stall detection + recovery routing."""
from src.two_tier_graph.stall_detection import detect_stall, StallSignal


def test_repeated_action_no_progress():
    """3 consecutive same-action same-arg + no step growth → stall."""
    history = [
        {"action_type": "explore_frontier", "args": {"frontier_idx": 14}},
        {"action_type": "explore_frontier", "args": {"frontier_idx": 14}},
        {"action_type": "explore_frontier", "args": {"frontier_idx": 14}},
    ]
    signal = detect_stall(history, steps_taken=0)
    assert signal is not None
    assert signal.kind == "repeated_action_no_progress"
    assert signal.repeated_count == 3


def test_no_stall_when_progressing():
    """Steps growing → no stall even if same action."""
    history = [
        {"action_type": "explore_frontier", "args": {"frontier_idx": 14}},
        {"action_type": "explore_frontier", "args": {"frontier_idx": 14}},
    ]
    signal = detect_stall(history, steps_taken=5)
    assert signal is None


def test_no_stall_when_different_actions():
    """Different actions → no stall."""
    history = [
        {"action_type": "explore_frontier", "args": {"frontier_idx": 14}},
        {"action_type": "explore_seed", "args": {"seed_id": "desk"}},
        {"action_type": "explore_frontier", "args": {"frontier_idx": 15}},
    ]
    signal = detect_stall(history, steps_taken=0)
    assert signal is None


def test_stall_recovery_routing():
    """after_memory should route to stall_recovery when stall_signal present."""
    from src.two_tier_graph.edges import after_memory
    state = {"last_transition": {"reason": "continue"},
             "stall_signal": {"kind": "repeated_action_no_progress"}}
    assert after_memory(state) == "stall_recovery"


def test_no_stall_recovery_when_no_signal():
    """after_memory should route normally when no stall_signal."""
    from src.two_tier_graph.edges import after_memory
    state = {"last_transition": {"reason": "continue"}}
    assert after_memory(state) == "continue"
