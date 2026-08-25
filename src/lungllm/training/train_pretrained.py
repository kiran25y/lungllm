# REPO PATH: src/lungllm/training/train_pretrained.py
"""AST transfer learning (linear-probe or --finetune) for 4-class anomaly."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from lungllm.data.features import load_audio
from lungllm.data.dataset import EVENT_TO_IDX
from lungllm.models.pretrained_encoder import ASTEncoder, PretrainedAnomalyModel
from lungllm.training.engine import icbhi_score


class WaveDataset(Dataset):
    def __init__(self, manifest, sr=16000, maxs=10.0):
        p = Path(manifest)
        if not p.exists() and p.with_suffix(".csv").exists(): p = p.with_suffix(".csv")
        self.rows = (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)).to_dict("records")
        self.sr = sr; self.mx = int(maxs*sr); self.mn = int(0.5*sr)
    def __len__(self): return len(self.rows)
    def labels(self): return [EVENT_TO_IDX.get(r.get("event"), 0) for r in self.rows]
    def __getitem__(self, i):
        r = self.rows[i]; w = load_audio(r["audio_path"], self.sr)
        s, e = int(float(r["start"])*self.sr), int(float(r["end"])*self.sr)
        if e > s: w = w[s:e]
        if w.numel() > self.mx: w = w[:self.mx]
        if w.numel() < self.mn: w = torch.nn.functional.pad(w, (0, self.mn-w.numel()))
        return w, EVENT_TO_IDX.get(r.get("event"), 0)


def collate(b): return [x[0] for x in b], torch.tensor([x[1] for x in b], dtype=torch.long)


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); preds, gts = [], []
    for w, l in dl: preds.append(model(w)["logits"].argmax(-1).cpu()); gts.append(l)
    preds, gts = torch.cat(preds), torch.cat(gts); se, sp, sc = icbhi_score(preds, gts)
    pc = {c: (((preds==gts)&(gts==c)).sum().item()/max(int((gts==c).sum()),1) if int((gts==c).sum()) else None) for c in range(4)}
    return {"acc":(preds==gts).float().mean().item(),"se":se,"sp":sp,"icbhi":sc,"pc":pc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train.parquet")
    ap.add_argument("--val", default="data/splits/val.parquet")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--num_workers", type=int, default=0); ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/sft/ast.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable — check CUDA_VISIBLE_DEVICES.")
    tr, va = WaveDataset(a.train), WaveDataset(a.val)
    if a.balanced_sampler:
        lab = np.array(tr.labels()); cnt = np.bincount(lab, minlength=4).astype(float); cnt[cnt==0]=1
        sm = WeightedRandomSampler(torch.as_tensor(1.0/cnt[lab], dtype=torch.double), len(lab), True)
        tdl = DataLoader(tr, batch_size=a.batch_size, sampler=sm, collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    else:
        tdl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    vdl = DataLoader(va, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=a.num_workers)
    model = PretrainedAnomalyModel(ASTEncoder(a.model_name, freeze=not a.finetune)).to(a.device)
    params = [p for p in model.parameters() if p.requires_grad]; nb = len(tdl)
    print(f"[ast] {'FINE-TUNE' if a.finetune else 'LINEAR-PROBE'} | trainable {sum(p.numel() for p in params):,} | {nb} batches/epoch", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); best = -1.0
    for ep in range(1, a.epochs+1):
        model.train(); run=0.0; n=0; t0=time.time()
        for bi,(w,l) in enumerate(tdl,1):
            l=l.to(a.device); loss=torch.nn.functional.cross_entropy(model(w)["logits"],l)
            opt.zero_grad(); loss.backward(); opt.step(); run+=loss.item()*l.size(0); n+=l.size(0)
            if bi%a.log_every==0: print(f"  ep{ep} {bi}/{nb} loss {run/max(n,1):.3f} {(time.time()-t0)/bi:.2f}s/b", flush=True)
        m=evaluate(model,vdl); pc=m["pc"]
        print(f"epoch {ep:02d} loss {run/max(n,1):.4f} acc {m['acc']:.4f} ICBHI {m['icbhi']:.4f} (Se {m['se']:.3f} Sp {m['sp']:.3f}) NWCB {pc[0]} {pc[1]} {pc[2]} {pc[3]}", flush=True)
        if m["icbhi"]>best: best=m["icbhi"]; torch.save({"model":model.state_dict(),"epoch":ep,"metrics":m}, a.out); print("  -> saved best", round(best,4), flush=True)
    print("done best val ICBHI", round(best,4), flush=True)


if __name__ == "__main__":
    main()
