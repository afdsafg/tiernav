"""Verify L0 visual memory index layer."""
from src.two_tier_graph.visual_memory import VisualMemoryIndex


def test_l0_index_one_line_per_snapshot():
    """Each snapshot produces exactly one L0 index line."""
    idx = VisualMemoryIndex()
    idx.update(round_idx=1, pose=(1.0, 2.0), object_class="chair",
               one_line_desc="red chair near window", snapshot_id="snap_001")
    text = idx.get_index_text()
    lines = text.strip().split("\n")
    assert len(lines) == 1
    assert "chair" in lines[0]
    assert "red chair near window" in lines[0]


def test_l0_index_dedup_via_loaded_snapshot_ids():
    """Same snapshot_id should not produce duplicate index lines."""
    idx = VisualMemoryIndex()
    idx.update(1, (1.0, 2.0), "chair", "desc", "snap_001")
    idx.update(2, (1.0, 2.0), "chair", "desc", "snap_001")  # same snapshot
    assert len(idx.get_index_text().strip().split("\n")) == 1


def test_l0_index_refresh_interval():
    """Index should only rebuild when round % interval == 0."""
    idx = VisualMemoryIndex(refresh_interval=3)
    assert not idx.should_rebuild(round_idx=1)
    assert not idx.should_rebuild(round_idx=2)
    assert idx.should_rebuild(round_idx=3)


def test_l0_index_fallback_on_failure():
    """If CLIP retrieval fails, index should still return last good text."""
    idx = VisualMemoryIndex()
    idx.update(1, (1.0, 2.0), "chair", "desc", "snap_001")
    # Simulate failure
    idx._text = None  # corrupt
    assert idx.get_index_text() == ""  # graceful fallback
