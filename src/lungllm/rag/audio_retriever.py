# REPO PATH: src/lungllm/rag/audio_retriever.py
"""Retrieve acoustically-similar past cases (their reports) for a query waveform."""
from __future__ import annotations
import numpy as np, torch


class AudioRetriever:
    def __init__(self, index_path, encoder, top_k=5, device="cpu"):
        import faiss
        self.index = faiss.read_index(str(index_path))
        self.meta = np.load(str(index_path) + ".meta.npy", allow_pickle=True)
        self.enc = encoder.to(device).eval(); self.device = device; self.top_k = top_k
        self._faiss = faiss

    @torch.no_grad()
    def retrieve(self, waveform, top_k=None):
        k = top_k or self.top_k
        z = self.enc([waveform])["clip_embedding"].cpu().numpy().astype("float32")
        self._faiss.normalize_L2(z)
        scores, idx = self.index.search(z, k)
        return [{"report": self.meta[i].get("report"), "event": self.meta[i].get("event"),
                 "diagnosis": self.meta[i].get("diagnosis"), "score": float(s)}
                for i, s in zip(idx[0], scores[0])]