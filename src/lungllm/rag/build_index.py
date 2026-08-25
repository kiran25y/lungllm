# REPO PATH: src/lungllm/rag/build_index.py
"""Phase 4 — build a FAISS index over report text embeddings (JINA/sentence-transformers)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data/rag/reports/reports.jsonl")
    ap.add_argument("--out", default="data/rag/index/reports.faiss")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    import faiss
    from sentence_transformers import SentenceTransformer
    rows = [json.loads(l) for l in Path(a.reports).read_text().splitlines() if l.strip()]
    model = SentenceTransformer(a.embed_model)
    emb = model.encode([r["report"] for r in rows], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    np.save(str(out) + ".meta.npy", np.array([r["report"] for r in rows], dtype=object))
    print(f"indexed {len(rows)} reports (dim {emb.shape[1]}) ->", out)


if __name__ == "__main__":
    main()
