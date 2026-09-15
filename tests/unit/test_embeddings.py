"""Unit tests for Embedding Cache."""

import os
from app.knowledge.embeddings import EmbeddingCache

def test_embedding_cache_add_and_search(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(cache_dir=str(cache_dir))
    
    # Needs a real sentence transformer, so this will download the small model
    # To keep tests fast, we'll just test if it doesn't crash on standard operations
    
    doc = "The primary key of the customer table is user_id. Phone number is PII."
    cache.add_document(doc, source="doc.txt")
    
    # Should have saved index
    assert os.path.exists(cache_dir / "faiss.index")
    assert os.path.exists(cache_dir / "metadata.json")
    
    # Search
    results = cache.search("What is the primary key?", top_k=1)
    
    assert len(results) == 1
    assert "user_id" in results[0]["text"]
    assert results[0]["source"] == "doc.txt"
