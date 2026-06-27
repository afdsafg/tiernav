"""Verify CaptionStore disk caching + retrieval."""
import tempfile
import os
from src.two_tier_graph.visual_memory import CaptionStore


def test_caption_store_write_read():
    """Caption written to disk can be read back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaptionStore(cache_dir=tmpdir)
        store.put("snap_001", "A red chair near the window")
        assert store.get("snap_001") == "A red chair near the window"
        # Verify on disk
        assert os.path.exists(os.path.join(tmpdir, "snap_001.txt"))


def test_caption_store_get_missing_returns_none():
    """Missing caption returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaptionStore(cache_dir=tmpdir)
        assert store.get("snap_999") is None


def test_caption_store_disk_persistence():
    """Caption survives store recreation (re-read from disk)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store1 = CaptionStore(cache_dir=tmpdir)
        store1.put("snap_001", "blue sofa")
        # New store instance, same dir
        store2 = CaptionStore(cache_dir=tmpdir)
        assert store2.get("snap_001") == "blue sofa"


def test_caption_store_top_k_by_clip():
    """top_k should return captions ranked by CLIP similarity (stub: returns all)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaptionStore(cache_dir=tmpdir)
        store.put("snap_001", "red chair")
        store.put("snap_002", "blue sofa")
        store.put("snap_003", "green plant")
        # Without real CLIP, returns all captions up to k
        results = store.top_k(query="chair", k=2)
        assert len(results) <= 2
