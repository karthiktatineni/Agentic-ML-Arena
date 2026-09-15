"""Knowledge Retriever (PRD v2 Section 7)."""

import os
from typing import List, Dict, Any
from app.knowledge.ingest import DocumentIngestor
from app.knowledge.pii_screen import TextPIIScanner
from app.knowledge.embeddings import EmbeddingCache

class KnowledgeRetriever:
    """Manages document ingestion, screening, and retrieval."""
    
    def __init__(self, cache_dir: str = ".knowledge"):
        self.cache = EmbeddingCache(cache_dir=cache_dir)
        
    def ingest_directory(self, directory: str):
        """Ingest all supported documents from a directory."""
        if not os.path.isdir(directory):
            return
            
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath):
                continue
                
            try:
                # 1. Ingest
                raw_text = DocumentIngestor.parse_file(filepath)
                # 2. Scrub PII
                safe_text = TextPIIScanner.scrub_text(raw_text)
                # 3. Embed & Cache
                self.cache.add_document(safe_text, source=filename)
            except Exception as e:
                print(f"Failed to ingest {filename}: {e}")
                
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge chunks for a query."""
        return self.cache.search(query, top_k=top_k)
