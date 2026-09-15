"""Embeddings & Vector Store Cache (PRD v2 Section 7)."""

import os
import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

class EmbeddingCache:
    """Caches document embeddings locally using FAISS."""
    
    def __init__(self, cache_dir: str = ".knowledge", model_name: str = "all-MiniLM-L6-v2"):
        self.cache_dir = cache_dir
        self.index_path = os.path.join(cache_dir, "faiss.index")
        self.meta_path = os.path.join(cache_dir, "metadata.json")
        self.model_name = model_name
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.model = None
        self.index = None
        self.metadata = []
        
        self._load_index()
        
    def _lazy_init_model(self):
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for EmbeddingCache")
            self.model = SentenceTransformer(self.model_name)
            
    def _load_index(self):
        if faiss is None:
            raise ImportError("faiss-cpu is required for EmbeddingCache")
            
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            # Dimension for all-MiniLM-L6-v2 is 384
            self.index = faiss.IndexFlatIP(384) # Inner product for cosine similarity (if normalized)
            self.metadata = []
            
    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Simple character-based chunking."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
        
    def add_document(self, text: str, source: str = "unknown"):
        """Embed and store a document in the FAISS index."""
        # Check if already cached by hash
        doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if any(m.get("doc_hash") == doc_hash for m in self.metadata):
            return # Already cached
            
        self._lazy_init_model()
        chunks = self._chunk_text(text)
        if not chunks:
            return
            
        # Generate embeddings
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        # Add to FAISS
        start_id = len(self.metadata)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": start_id + i,
                "doc_hash": doc_hash,
                "source": source,
                "text": chunk
            })
            
        self._save_index()
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the FAISS index for similar chunks."""
        if not self.metadata or self.index.ntotal == 0:
            return []
            
        self._lazy_init_model()
        query_emb = self.model.encode([query], normalize_embeddings=True)
        
        distances, indices = self.index.search(np.array(query_emb).astype('float32'), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                res["score"] = float(dist)
                results.append(res)
                
        return results
