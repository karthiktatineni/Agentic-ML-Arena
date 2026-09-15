"""Unit tests for Knowledge Retriever."""

import os
from app.knowledge.retriever import KnowledgeRetriever

def test_knowledge_retriever(tmp_path):
    # Setup dummy directory
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    
    (doc_dir / "rules.md").write_text("The threshold must be at least 0.5. Email me at admin@company.com.")
    (doc_dir / "guide.txt").write_text("Use random forest for baseline.")
    
    cache_dir = tmp_path / ".knowledge"
    
    retriever = KnowledgeRetriever(cache_dir=str(cache_dir))
    retriever.ingest_directory(str(doc_dir))
    
    # Should scrub the email
    results = retriever.retrieve("What is the threshold?", top_k=2)
    assert len(results) > 0
    assert "threshold must be at least 0.5" in results[0]["text"]
    assert "admin@company.com" not in results[0]["text"]
    assert "[EMAIL_REDACTED]" in results[0]["text"]
    
    results2 = retriever.retrieve("baseline model", top_k=1)
    assert "random forest" in results2[0]["text"]
