# REPO PATH: src/lungllm/training/train_sft_gen.py
"""Generative SFT: teach audio -> report (bridge + LLM LoRA) on the clean report corpus."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from lungllm.data.features import load_audio
from lungllm.models.gen_model import GenModel


class ReportSet(Dataset):
    def __init__(self, jsonl, sr=16000, maxs=10.0):
        self.rows = [json.loads(l) for l in Path(jsonl).read_text().splitlines() if l.strip()]
        self.sr = sr; self.mx = int(maxs * sr); self.mn = int(0.5 * sr)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; w = load_audio(r["audio_path"], self.sr)
        if w.numel() > self.mx: w = w[:self.mx]
        if w.numel() < self.mn: w = torch.nn.functional.pad(w, (0, self.mn - w.numel()))
        return w, r["report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data/rag/reports/reports_train.jsonl")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--encoder_ckpt", default="checkpoints/aligned/ast_aligned_clean.pt")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--num_workers", type=int, default=0); ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/gen/sft_gen.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    model = GenModel(a.llm, encoder_ckpt=a.encoder_ckpt if Path(a.encoder_ckpt).exists() else None).to(a.device)
    tok = model.llm.tok

    def collate(b):
        waves = [x[0] for x in b]; reports = [x[1] for x in b]
        enc = tok(reports, return_tensors="pt", padding=True, truncation=True, max_length=a.max_len)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        return waves, enc["input_ids"], enc["attention_mask"], labels

    dl = DataLoader(ReportSet(a.reports), batch_size=a.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]; nb = len(dl)
    print(f"[gen-sft] trainable {sum(p.numel() for p in params):,} | {nb} batches/epoch", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0; n = 0; t0 = time.time()
        for bi, (waves, ids, mask, labels) in enumerate(dl, 1):
            ids, mask, labels = ids.to(a.device), mask.to(a.device), labels.to(a.device)
            loss = model(waves, ids, mask, labels).loss
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item(); n += 1
            if bi % a.log_every == 0:
                print(f"  ep{ep} {bi}/{nb} loss {run/max(n,1):.3f} {(time.time()-t0)/bi:.2f}s/b", flush=True)
        print(f"epoch {ep:02d} loss {run/max(n,1):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, a.out)
    print("saved generative SFT model ->", a.out, flush=True)


if __name__ == "__main__":
    main()