"""Verify L2 image recall layer — token-budgeted snapshot recall."""
import tempfile
from src.two_tier_graph.visual_memory import ImageRecallStore


def test_image_recall_store_token_budget():
    """Store should respect token budget when selecting images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ImageRecallStore(cache_dir=tmpdir, token_budget=3000, tokens_per_image=1000)
        # 5 images available, budget allows 3
        for i in range(5):
            store.register(f"snap_{i}", image_b64=f"fake_b64_{i}", tokens=1000)
        selected = store.select_for_recall(query="chair", loaded_ids=set())
        assert len(selected) <= 3  # budget caps at 3


def test_image_recall_lru_dedup():
    """Already-loaded snapshots should not be recalled again."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ImageRecallStore(cache_dir=tmpdir, token_budget=3000, tokens_per_image=1000)
        store.register("snap_0", image_b64="b64_0", tokens=1000)
        store.register("snap_1", image_b64="b64_1", tokens=1000)
        # snap_0 already loaded
        selected = store.select_for_recall(query="chair", loaded_ids={"snap_0"})
        assert all(s["snapshot_id"] != "snap_0" for s in selected)


def test_image_recall_empty_when_no_images():
    """No registered images → empty selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ImageRecallStore(cache_dir=tmpdir, token_budget=3000, tokens_per_image=1000)
        selected = store.select_for_recall(query="chair", loaded_ids=set())
        assert selected == []


def test_need_visual_recall_field():
    """State should have need_visual_recall field."""
    from src.two_tier_graph.state import TwoTierState
    assert "need_visual_recall" in TwoTierState.__annotations__
