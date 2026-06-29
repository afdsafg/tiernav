"""Semantic memory store — LlamaIndex retrieval with keyword fallback.

Borrowed from Claude Code findRelevantMemories. When LlamaIndex is installed
(in langgraph conda env), uses vector retriever over CLIP embeddings.
Otherwise falls back to keyword matching (current agent_memory.py:62 behavior).
"""
import logging
from typing import Optional


class SemanticMemoryStore:
    """Retrieval backend for visual memory.

    engine="llamaindex": uses LlamaIndex VectorStoreIndex (lazy import)
    engine="keyword": uses simple keyword matching (fallback, current behavior)
    """
    def __init__(self, engine: str = "keyword"):
        self.engine = engine
        self._documents: list[dict] = []  # [{"id": str, "text": str, "metadata": dict}]
        self._llamaindex_available = False
        self._index = None

        if engine == "llamaindex":
            try:
                from llama_index.core import VectorStoreIndex, Document
                self._llamaindex_available = True
                self._index = VectorStoreIndex([])
                self._Document = Document
            except ImportError:
                logging.warning("llama_index not installed, falling back to keyword engine")
                self.engine = "keyword"

    def add(self, doc_id: str, text: str, metadata: dict):
        """Add a document to the store."""
        doc = {"id": doc_id, "text": text, "metadata": metadata}
        self._documents.append(doc)
        if self._llamaindex_available and self._index:
            from llama_index.core import Document
            self._index.insert_doc(Document(text=text, metadata=metadata))

    def query(self, query: str, top_k: int = 3) -> list[dict]:
        """Query the store for top-K matching documents."""
        if self._llamaindex_available and self._index:
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            return [{"id": n.node_id, "text": n.text, "metadata": n.metadata} for n in nodes]
        else:
            # Keyword fallback (current agent_memory.py:62 behavior)
            return self._keyword_query(query, top_k)

    def _keyword_query(self, query: str, top_k: int) -> list[dict]:
        """Simple keyword matching fallback."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self._documents:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]
