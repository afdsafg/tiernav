"""Tests for dump_context_tokens CLI token-analysis table."""
import json

from src.tiernav_runtime.cli import dump_context_tokens


def _write_round(path, sections):
    entry = {"round": 0, "step": 0, "sections": sections}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _section(name, tokens, *, cacheable, cache_break=False):
    return {"name": name, "tokens": tokens, "cacheable": cacheable, "cache_break": cache_break}


def test_dump_context_tokens_existing_file(tmp_path):
    audit_dir = tmp_path / "prompt_audit"
    audit_dir.mkdir()
    path = audit_dir / "test_ep.jsonl"

    round0 = [
        _section("task_instruction", 100, cacheable=True),
        _section("action_schema", 50, cacheable=True),
        _section("task_state", 30, cacheable=False, cache_break=True),
        _section("recent_trace", 20, cacheable=False),
    ]
    round1 = [
        _section("task_instruction", 120, cacheable=True),
        _section("action_schema", 50, cacheable=True),
        _section("task_state", 40, cacheable=False, cache_break=True),
        _section("recent_trace", 60, cacheable=False),
    ]
    _write_round(path, round0)
    _write_round(path, round1)

    result = dump_context_tokens("test_ep", str(tmp_path))

    assert "task_instruction" in result
    assert "task_state" in result
    assert "<- boundary" in result
    assert "cacheable" in result
    assert "yes" in result
    assert "no" in result
    assert "total avg tokens" in result


def test_dump_context_tokens_missing_file(tmp_path):
    result = dump_context_tokens("nonexistent", str(tmp_path))
    assert "no prompt audit log" in result
