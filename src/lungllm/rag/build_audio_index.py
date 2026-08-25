# REPO PATH: src/lungllm/rag/build_audio_index.py
"""Stronger RAG: FAISS index of AUDIO embeddings (aligned encoder) so retrieval works on
SOUND, not a text query. Each clip stores its report/event/diagnosis for retrieval."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from lungllm.data.features import load_audio
from lungllm.models.pretrained_encoder import ASTEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data/rag/reports/reports_train.jsonl")
    ap.add_argument("--encoder_ckpt", default="checkpoints/aligned/ast_aligned_clean.pt")
    ap.add_argument("--out", default="data/rag/index/audio.faiss")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import faiss
    enc = ASTEncoder(freeze=True)
    if Path(a.encoder_ckpt).exists():
        ck = torch.load(a.encoder_ckpt, map_location="cpu")
        enc.encoder.load_state_dict(ck.get("encoder", ck), strict=False) if hasattr(enc, "encoder") else enc.load_state_dict(ck.get("encoder", ck), strict=False)
    enc.to(a.device).eval()
    rows = [json.loads(l) for l in Path(a.reports).read_text().splitlines() if l.strip()]
    embs, meta = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            batch = rows[i:i+16]
            waves = []
            for r in batch:
                w = load_audio(r["audio_path"], 16000)
                if w.numel() > 160000: w = w[:160000]
                if w.numel() < 8000: w = torch.nn.functional.pad(w, (0, 8000-w.numel()))
                waves.append(w)
            z = enc(waves)["clip_embedding"].cpu().numpy()
            embs.append(z); meta.extend(batch)
            if i % 800 == 0: print(f"[audio-index] {i}/{len(rows)}", flush=True)
    embs = np.concatenate(embs).astype("float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    np.save(str(out) + ".meta.npy", np.array(meta, dtype=object))
    print(f"[audio-index] {len(meta)} clips (dim {embs.shape[1]}) -> {out}", flush=True)


if __name__ == "__main__":
    main()