"""Tests for PromptAuditRecorder per-round prompt-section JSONL output."""
import json

from src.tiernav_runtime.contracts import ContextSection
from src.tiernav_runtime.recorder import PromptAuditRecorder


def _section(name: str, content: str, *, cacheable: bool, cache_break: bool = False) -> ContextSection:
    return ContextSection(
        name=name,
        content=content,
        cacheable=cacheable,
        cache_break=cache_break,
        token_estimate=len(content),
    )


def test_prompt_audit_records_two_rounds(tmp_path):
    recorder = PromptAuditRecorder(tmp_path)
    sections_r0 = [
        _section("system", "system prompt", cacheable=True),
        _section("task", "what is on the table?", cacheable=False, cache_break=True),
    ]
    sections_r1 = [
        _section("system", "system prompt", cacheable=True),
        _section("task", "what is on the table?", cacheable=False, cache_break=True),
        _section("obs", "saw a chair", cacheable=False),
    ]

    recorder.record("ep-1", 0, 0, sections_r0)
    recorder.record("ep-1", 1, 1, sections_r1)

    path = tmp_path / "prompt_audit" / "ep-1.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    for i, line in enumerate(lines):
        entry = json.loads(line)
        assert set(entry.keys()) >= {"round", "step", "sections"}
        assert entry["round"] == i
        assert entry["step"] == i
        sections = entry["sections"]
        assert isinstance(sections, list)
        assert len(sections) == len([sections_r0, sections_r1][i])
        for s in sections:
            assert "content" in s
            assert "cache_break" in s
            assert "cacheable" in s
            assert "name" in s
            assert "hash" in s
            assert "tokens" in s
        # Verify full content round-trips
        names = [s["name"] for s in sections]
        assert "system" in names


def test_prompt_audit_creates_dir(tmp_path):
    recorder = PromptAuditRecorder(tmp_path)
    assert (tmp_path / "prompt_audit").is_dir()


def test_prompt_audit_appends_across_calls(tmp_path):
    recorder = PromptAuditRecorder(tmp_path)
    s = _section("x", "y", cacheable=True)
    recorder.record("ep-2", 0, 0, [s])
    recorder.record("ep-2", 1, 0, [s])
    recorder.record("ep-2", 2, 1, [s])

    path = tmp_path / "prompt_audit" / "ep-2.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    rounds = [json.loads(l)["round"] for l in lines]
    assert rounds == [0, 1, 2]


def test_prompt_audit_cache_break_field(tmp_path):
    recorder = PromptAuditRecorder(tmp_path)
    sections = [
        _section("stable", "stable", cacheable=True, cache_break=False),
        _section("unstable", "changes", cacheable=False, cache_break=True),
    ]
    recorder.record("ep-3", 0, 0, sections)

    path = tmp_path / "prompt_audit" / "ep-3.jsonl"
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["sections"][0]["cache_break"] is False
    assert entry["sections"][1]["cache_break"] is True
