"""Verify prompt section registry + cache boundary."""
from src.two_tier_graph.prompt_sections import PromptSection, build_planner_prompt


def test_prompt_sections_ordered():
    sections = build_planner_prompt(state={}, resources=None)
    names = [s.name for s in sections]
    assert "task_instruction" in names
    assert "action_schema" in names
    assert "memory_index" in names
    assert len(sections) >= 5


def test_cacheable_sections_come_first():
    """All cacheable=True sections must precede cacheable=False for cache boundary."""
    sections = build_planner_prompt(state={}, resources=None)
    seen_non_cacheable = False
    for s in sections:
        if not s.cacheable:
            seen_non_cacheable = True
        elif seen_non_cacheable:
            assert False, f"Cacheable section {s.name} after non-cacheable"


def test_cache_boundary_marked():
    """build_planner_prompt should mark where cache boundary goes."""
    sections = build_planner_prompt(state={}, resources=None)
    # At least one cacheable and one non-cacheable
    assert any(s.cacheable for s in sections)
    assert any(not s.cacheable for s in sections)


def test_memory_index_included():
    """L0 index text should be in memory_index section when present."""
    state = {"l0_index_text": "[R1, pose=(1,2), obj=chair] red chair"}
    sections = build_planner_prompt(state=state, resources=None)
    mem_section = [s for s in sections if s.name == "memory_index"][0]
    assert "chair" in mem_section.content
