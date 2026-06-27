"""Verify shared helpers extracted from agent_workflow.py (D5 dedup)."""
from src.agent_context import ContextManager
from src.shared_helpers import (
    _NAV_OBJ_INVALID,
    _is_valid_object_desc,
    _build_messages,
)


def test_nav_obj_invalid_is_set():
    assert isinstance(_NAV_OBJ_INVALID, (set, frozenset))
    assert len(_NAV_OBJ_INVALID) > 0
    # representative members preserved verbatim
    assert "forward" in _NAV_OBJ_INVALID
    assert "kitchen" in _NAV_OBJ_INVALID


def test_is_valid_object_desc_rejects_invalid():
    assert not _is_valid_object_desc("")
    assert not _is_valid_object_desc("forward")
    assert not _is_valid_object_desc("kitchen")
    assert not _is_valid_object_desc("room 1")
    assert not _is_valid_object_desc("42")
    assert not _is_valid_object_desc("a")  # len < 2


def test_is_valid_object_desc_accepts_valid():
    assert _is_valid_object_desc("chair")
    assert _is_valid_object_desc("red mug")
    assert _is_valid_object_desc("wooden table")


def test_build_messages_system_plus_context():
    ctx = ContextManager()
    ctx.stage_messages = [{"role": "user", "content": "hello"}]
    msgs = _build_messages(ctx, system_prompt="You are agent")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are agent"
    assert msgs[1]["role"] == "user"


def test_build_messages_includes_transitions():
    from src.agent_context import StageTransition
    ctx = ContextManager()
    ctx.current_stage = 2
    ctx.transitions.append(StageTransition(from_stage=1, to_stage=2, summary="did X"))
    ctx.stage_messages = [{"role": "user", "content": "q"}]
    msgs = _build_messages(ctx, system_prompt="sys")
    # system + transition summary (assistant) + stage user msg
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert "Stage 1→2 summary" in msgs[1]["content"]
