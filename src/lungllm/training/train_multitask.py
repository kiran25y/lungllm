# REPO PATH: src/lungllm/training/train_multitask.py
"""Multi-task AST transfer: anomaly head + disease head (disease labels from ICBHI+Mendeley)."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from lungllm.data.features import load_audio
from lungllm.data.dataset import EVENT_TO_IDX
from lungllm.models.pretrained_encoder import ASTEncoder, PretrainedMultiTaskModel
from lungllm.training.engine import icbhi_score

IGNORE = -100
DCANON = {"healthy":"Healthy","normal":"Healthy","n":"Healthy","copd":"COPD","asthma":"Asthma",
 "bronchiectasis":"Bronchiectasis","bron":"Bronchiectasis","bronchiolitis":"Bronchiolitis",
 "pneumonia":"Pneumonia","urti":"URTI","lrti":"LRTI","heart failure":"Heart Failure",
 "lung fibrosis":"Lung Fibrosis","fibrosis":"Lung Fibrosis","pleural effusion":"Pleural Effusion",
 "plueral effusion":"Pleural Effusion"}


def canon(x):
    if x is None: return None
    try:
        if isinstance(x, float) and np.isnan(x): return None
    except Exception: pass
    k = str(x).strip().lower()
    if not k or k == "nan": return None
    return DCANON.get(k, str(x).strip().title())


def read_rows(p):
    p = str(p); return (pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)).to_dict("records")


class DS(Dataset):
    def __init__(self, manifest, dv, sr=16000, maxs=10.0):
        self.rows = read_rows(manifest); self.dv = dv; self.sr = sr
        self.mx = int(maxs*sr); self.mn = int(0.5*sr)
    def __len__(self): return len(self.rows)
    def alabels(self): return [EVENT_TO_IDX.get(r.get("event"), 0) for r in self.rows]
    def __getitem__(self, i):
        r = self.rows[i]; w = load_audio(r["audio_path"], self.sr)
        s, e = int(float(r["start"])*self.sr), int(float(r["end"])*self.sr)
        if e > s: w = w[s:e]
        if w.numel() > self.mx: w = w[:self.mx]
        if w.numel() < self.mn: w = torch.nn.functional.pad(w, (0, self.mn-w.numel()))
        d = canon(r.get("diagnosis")); di = self.dv.get(d, IGNORE) if d is not None else IGNORE
        return w, EVENT_TO_IDX.get(r.get("event"), 0), di


def collate(b):
    return [x[0] for x in b], torch.tensor([x[1] for x in b]), torch.tensor([x[2] for x in b])


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); ap=[]; ag=[]; dp=[]; dg=[]
    for w, al, dl_ in dl:
        o = model(w); ap.append(o["anomaly"].argmax(-1).cpu()); ag.append(al)
        if "disease" in o:
            m = dl_ != IGNORE
            if m.any(): dp.append(o["disease"].argmax(-1).cpu()[m]); dg.append(dl_[m])
    ap = torch.cat(ap); ag = torch.cat(ag); se, sp, sc = icbhi_score(ap, ag)
    pc = {c: (((ap==ag)&(ag==c)).sum().item()/max(int((ag==c).sum()),1) if int((ag==c).sum()) else None) for c in range(4)}
    da = None
    if dg: dp = torch.cat(dp); dg = torch.cat(dg); da = (dp==dg).float().mean().item()
    return {"acc":(ap==ag).float().mean().item(),"se":se,"sp":sp,"icbhi":sc,"pc":pc,"da":da}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train_all.parquet")
    ap.add_argument("--val", default="data/splits/val.parquet")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--disease_weight", type=float, default=0.3)
    ap.add_argument("--finetune", action="store_true"); ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--num_workers", type=int, default=0); ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/sft/ast_multitask.pt")
    ap.add_argument("--init_encoder", default=None, help="aligned encoder ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable — check CUDA_VISIBLE_DEVICES.")
    dv = {d: i for i, d in enumerate(sorted({canon(r.get("diagnosis")) for r in read_rows(a.train)} - {None}))}
    print("[multitask] disease classes", len(dv), list(dv), flush=True)
    tr, va = DS(a.train, dv), DS(a.val, dv)
    if a.balanced_sampler:
        lab = np.array(tr.alabels()); cnt = np.bincount(lab, minlength=4).astype(float); cnt[cnt==0]=1
        sm = WeightedRandomSampler(torch.as_tensor(1.0/cnt[lab], dtype=torch.double), len(lab), True)
        tdl = DataLoader(tr, batch_size=a.batch_size, sampler=sm, collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    else:
        tdl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    vdl = DataLoader(va, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=a.num_workers)
    enc = ASTEncoder(a.model_name, freeze=not a.finetune)
    if a.init_encoder:
        ck = torch.load(a.init_encoder, map_location="cpu")
        enc.load_state_dict(ck.get("encoder", ck), strict=False)
        print("[multitask] loaded aligned encoder from", a.init_encoder, flush=True)
    model = PretrainedMultiTaskModel(enc, num_anomaly=4, num_disease=len(dv)).to(a.device)
    params = [p for p in model.parameters() if p.requires_grad]; nb = len(tdl)
    print(f"[multitask] {'FINE-TUNE' if a.finetune else 'LINEAR-PROBE'} | trainable {sum(p.numel() for p in params):,} | {nb} batches/epoch", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); best=-1.0
    for ep in range(1, a.epochs+1):
        model.train(); run=0.0; n=0; t0=time.time()
        for bi,(w,al,dlab) in enumerate(tdl,1):
            al=al.to(a.device); dlab=dlab.to(a.device); o=model(w)
            loss=F.cross_entropy(o["anomaly"],al)
            if (dlab!=IGNORE).any(): loss=loss+a.disease_weight*F.cross_entropy(o["disease"],dlab,ignore_index=IGNORE)
            opt.zero_grad(); loss.backward(); opt.step(); run+=loss.item()*al.size(0); n+=al.size(0)
            if bi%a.log_every==0: print(f"  ep{ep} {bi}/{nb} loss {run/max(n,1):.3f} {(time.time()-t0)/bi:.2f}s/b", flush=True)
        m=evaluate(model,vdl); pc=m["pc"]; ds=f" disease_acc {m['da']:.3f}" if m["da"] is not None else ""
        print(f"epoch {ep:02d} loss {run/max(n,1):.4f} acc {m['acc']:.4f} ICBHI {m['icbhi']:.4f} (Se {m['se']:.3f} Sp {m['sp']:.3f}) NWCB {pc[0]} {pc[1]} {pc[2]} {pc[3]}{ds}", flush=True)
        if m["icbhi"]>best: best=m["icbhi"]; torch.save({"model":model.state_dict(),"epoch":ep,"metrics":m,"disease_vocab":dv}, a.out); print("  -> saved best", round(best,4), flush=True)
    print("done best val ICBHI", round(best,4), flush=True)


if __name__ == "__main__":
    main()
