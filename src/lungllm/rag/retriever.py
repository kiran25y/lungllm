# REPO PATH: src/lungllm/rag/retriever.py
"""Phase 4 — dense retriever: top-k reports for a query string (or embedding)."""
from __future__ import annotations
import numpy as np


class FaissRetriever:
    def __init__(self, index_path, embed_model="sentence-transformers/all-MiniLM-L6-v2", top_k=5):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.index = faiss.read_index(str(index_path))
        self.reports = np.load(str(index_path) + ".meta.npy", allow_pickle=True)
        self.model = SentenceTransformer(embed_model)
        self.top_k = top_k

    def retrieve(self, query, top_k=None):
        k = top_k or self.top_k
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return [(self.reports[i], float(s)) for i, s in zip(idx[0], scores[0])]
