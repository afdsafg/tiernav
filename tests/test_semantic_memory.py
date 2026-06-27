"""Verify SemanticMemoryStore — LlamaIndex backend with keyword fallback."""
import pytest


def test_semantic_memory_store_exists():
    """SemanticMemoryStore class exists."""
    from src.two_tier_graph.semantic_memory import SemanticMemoryStore
    assert SemanticMemoryStore is not None


def test_semantic_memory_store_keyword_fallback():
    """When LlamaIndex not available, should fall back to keyword search."""
    from src.two_tier_graph.semantic_memory import SemanticMemoryStore
    store = SemanticMemoryStore(engine="keyword")
    store.add("snap_1", "red chair near window", {"pose": (1, 2)})
    store.add("snap_2", "blue sofa near door", {"pose": (3, 4)})
    results = store.query("chair", top_k=2)
    assert len(results) >= 1
    assert any("chair" in r["text"] for r in results)


def test_semantic_memory_store_llamaindex_lazy_import():
    """LlamaIndex engine should not crash on init if llama_index not installed."""
    from src.two_tier_graph.semantic_memory import SemanticMemoryStore
    # Should fall back to keyword if llama_index not installed
    store = SemanticMemoryStore(engine="llamaindex")
    store.add("snap_1", "red chair", {})
    # Query should still work (keyword fallback)
    results = store.query("chair", top_k=1)
    assert len(results) >= 1


def test_semantic_memory_store_add_and_query():
    """Add documents and query them."""
    from src.two_tier_graph.semantic_memory import SemanticMemoryStore
    store = SemanticMemoryStore(engine="keyword")
    store.add("doc1", "The kitchen has stainless steel appliances", {"room": "kitchen"})
    store.add("doc2", "The bedroom has a large window", {"room": "bedroom"})
    results = store.query("kitchen", top_k=1)
    assert len(results) == 1
    assert "kitchen" in results[0]["text"]
