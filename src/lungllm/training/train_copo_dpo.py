# REPO PATH: src/lungllm/training/train_copo_dpo.py
"""EXP4 — CoPO: acoustic-grounded Direct Preference Optimization.

Preference is grounded in ACOUSTIC correctness (not text similarity — the fix for
StethoLM's failed BERTScore-mDPO): for each audio with true event e,
  chosen   = a report describing e   (grounded)
  rejected = a report describing a different event (ungrounded).
DPO pushes the generative policy to prefer the acoustically-correct description over a
frozen reference (the SFT model). Requires the EXP3 generative SFT checkpoint.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lungllm.data.features import load_audio
from lungllm.models.gen_model import GenModel

EVENT_TEXT = {"normal": "Auscultation reveals no adventitious sounds.",
              "wheeze": "Auscultation reveals wheezing.",
              "crackle": "Auscultation reveals crackles.",
              "both": "Auscultation reveals crackles and wheezing."}
               
ALL = list(EVENT_TEXT.values())


class PrefSet(Dataset):
    def __init__(self, jsonl, sr=16000, maxs=10.0):
        self.rows = [json.loads(l) for l in Path(jsonl).read_text().splitlines()
                     if l.strip() and json.loads(l).get("event") in EVENT_TEXT]
        self.sr = sr; self.mx = int(maxs * sr); self.mn = int(0.5 * sr)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; w = load_audio(r["audio_path"], self.sr)
        if w.numel() > self.mx: w = w[:self.mx]
        if w.numel() < self.mn: w = torch.nn.functional.pad(w, (0, self.mn - w.numel()))
        ev = r["event"]; chosen = EVENT_TEXT[ev]
        rejected = next(t for t in ALL if t != chosen)
        return w, chosen, rejected


def seq_logp(model, waves, ids, mask, device):
    pref = model.audio_prefix(waves)
    out = model.llm(pref, ids.to(device), mask.to(device), labels=None)
    logits = out.logits
    B, k, _ = pref.shape; T = ids.size(1)
    lsm = torch.log_softmax(logits.float(), dim=-1)
    pred = lsm[:, k - 1:k - 1 + T, :]
    tok_lp = pred.gather(-1, ids.to(device).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask.to(device).float()).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data/rag/reports/reports_train.jsonl")
    ap.add_argument("--sft_ckpt", default="checkpoints/gen/sft_gen.pt")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--encoder_ckpt", default="checkpoints/aligned/ast_aligned_clean.pt")
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=24); ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/gen/copo.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    def build():
        m = GenModel(a.llm, encoder_ckpt=a.encoder_ckpt if Path(a.encoder_ckpt).exists() else None)
        if Path(a.sft_ckpt).exists():
            m.load_state_dict(torch.load(a.sft_ckpt, map_location="cpu")["model"], strict=False)
        return m.to(a.device)

    policy = build(); ref = build()
    for p in ref.parameters(): p.requires_grad_(False)
    ref.eval(); tok = policy.llm.tok

    def collate(b):
        waves = [x[0] for x in b]
        ch = tok([x[1] for x in b], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len)
        rj = tok([x[2] for x in b], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len)
        return waves, ch["input_ids"], ch["attention_mask"], rj["input_ids"], rj["attention_mask"]

    dl = DataLoader(PrefSet(a.reports), batch_size=a.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=a.lr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); nb = len(dl)
    print(f"[copo] {nb} batches/epoch | beta {a.beta}", flush=True)
    for ep in range(1, a.epochs + 1):
        policy.train(); run = 0.0; n = 0; acc = 0.0; t0 = time.time()
        for bi, (waves, ci, cm, ri, rm) in enumerate(dl, 1):
            lp_ch = seq_logp(policy, waves, ci, cm, a.device)
            lp_rj = seq_logp(policy, waves, ri, rm, a.device)
            with torch.no_grad():
                r_ch = seq_logp(ref, waves, ci, cm, a.device)
                r_rj = seq_logp(ref, waves, ri, rm, a.device)
            margin = a.beta * ((lp_ch - r_ch) - (lp_rj - r_rj))
            loss = -F.logsigmoid(margin).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); n += 1; acc += (margin > 0).float().mean().item()
            if bi % a.log_every == 0:
                print(f"  ep{ep} {bi}/{nb} loss {run/max(n,1):.3f} pref_acc {acc/max(n,1):.3f} {(time.time()-t0)/bi:.2f}s/b", flush=True)
        print(f"epoch {ep:02d} dpo_loss {run/max(n,1):.4f} pref_acc {acc/max(n,1):.3f}", flush=True)
        torch.save({"model": policy.state_dict(), "epoch": ep}, a.out)
    print("saved CoPO model ->", a.out, flush=True)


if __name__ == "__main__":
    main()