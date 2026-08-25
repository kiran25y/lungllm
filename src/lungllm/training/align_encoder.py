# REPO PATH: src/lungllm/training/align_encoder.py
"""Phase 1 — Stage-1 alignment trainer.

Pairs each recording with its clinical report string. A frozen medical-LLM text encoder
produces text embeddings (teacher); the audio encoder + projection heads are trained so
audio and text align (CKA + InfoNCE), regularized by masked-acoustic reconstruction.

Run:
  python -m lungllm.training.align_encoder \
     --reports data/rag/reports/reports.jsonl \
     --out checkpoints/aligned/ast_aligned.pt
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from lungllm.data.features import load_audio
from lungllm.models.pretrained_encoder import ASTEncoder
from lungllm.models.alignment.heads import ProjectionHead
from lungllm.models.alignment.losses import cka_loss, info_nce_loss


class ReportPairs(Dataset):
    """reports.jsonl: one JSON per line with {audio_path, report}."""
    def __init__(self, jsonl, sr=16000, max_seconds=10.0):
        self.rows = [json.loads(l) for l in Path(jsonl).read_text().splitlines() if l.strip()]
        self.sr = sr; self.mx = int(max_seconds * sr); self.mn = int(0.5 * sr)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; w = load_audio(r["audio_path"], self.sr)
        if w.numel() > self.mx: w = w[:self.mx]
        if w.numel() < self.mn: w = torch.nn.functional.pad(w, (0, self.mn - w.numel()))
        return w, r["report"]


def collate(b):
    return [x[0] for x in b], [x[1] for x in b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data/rag/reports/reports.jsonl")
    ap.add_argument("--text_model", default="google/medgemma-4b-it")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--lambda_cka", type=float, default=1.0)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/aligned/ast_aligned.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    # audio encoder (trainable) + text encoder (frozen teacher)
    enc = ASTEncoder(freeze=False).to(a.device)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.text_model)
    txt = AutoModel.from_pretrained(a.text_model).to(a.device).eval()
    for p in txt.parameters(): p.requires_grad_(False)
    text_dim = txt.config.hidden_size

    ph_a = ProjectionHead(enc.embed_dim, a.proj_dim).to(a.device)
    ph_t = ProjectionHead(text_dim, a.proj_dim).to(a.device)

    ds = ReportPairs(a.reports)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=0, drop_last=True)
    params = list(enc.parameters()) + list(ph_a.parameters()) + list(ph_t.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)

    @torch.no_grad()
    def embed_text(strings):
        t = tok(strings, return_tensors="pt", padding=True, truncation=True, max_length=64).to(a.device)
        out = txt(**t).last_hidden_state.mean(1)   # mean-pool
        return out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        enc.train(); run = 0.0; n = 0; t0 = time.time()
        for waves, reports in dl:
            za = ph_a(enc(waves)["clip_embedding"].to(a.device))
            zt = ph_t(embed_text(reports))
            loss = a.lambda_cka * cka_loss(za, zt) + a.lambda_nce * info_nce_loss(za, zt)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(reports); n += len(reports)
        print(f"epoch {ep:02d} | align_loss {run/max(n,1):.4f} | {(time.time()-t0):.0f}s", flush=True)
    torch.save({"encoder": enc.state_dict(), "proj_audio": ph_a.state_dict()}, a.out)
    print("saved aligned encoder ->", a.out)


if __name__ == "__main__":
    main()
