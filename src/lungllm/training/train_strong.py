"""Stronger anomaly trainer: Lungmix aug + cosine LR + class weights + label smoothing."""
from __future__ import annotations
import argparse, time, math, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from lungllm.data.features import load_audio
from lungllm.data.dataset import EVENT_TO_IDX
from lungllm.models.pretrained_encoder import ASTEncoder, PretrainedAnomalyModel
from lungllm.training.engine import icbhi_score

BITS = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}; INV = {v:k for k,v in BITS.items()}
def or_label(a,b):
    ca,wa=BITS[int(a)]; cb,wb=BITS[int(b)]; return INV[(ca|cb, wa|wb)]

def lungmix(xi,xj,yi,yj,alpha=1.0):
    n=min(xi.numel(),xj.numel()); xi,xj=xi[:n],xj[:n]
    lam=float(torch.distributions.Beta(alpha,alpha).sample())
    def loud(x):
        t=x.abs().mean()+2*x.abs().std(); return (x.abs()>t).float()
    ml=lam*torch.clamp(loud(xi)+loud(xj),max=1.0)
    rnd=(torch.rand(n)>0.5).float()
    mask=torch.clamp(ml+rnd*(1-(ml>0).float()),0,1)
    return mask*xi+(1-mask)*xj, or_label(yi,yj)

class WaveDS(Dataset):
    def __init__(self,manifest,sr=16000,maxs=10.0):
        p=Path(manifest)
        if not p.exists() and p.with_suffix(".csv").exists(): p=p.with_suffix(".csv")
        self.rows=(pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)).to_dict("records")
        self.sr=sr; self.mx=int(maxs*sr); self.mn=int(0.5*sr)
    def __len__(self): return len(self.rows)
    def labels(self): return [EVENT_TO_IDX.get(r.get("event"),0) for r in self.rows]
    def __getitem__(self,i):
        r=self.rows[i]; w=load_audio(r["audio_path"],self.sr)
        s,e=int(float(r["start"])*self.sr),int(float(r["end"])*self.sr)
        if e>s: w=w[s:e]
        if w.numel()>self.mx: w=w[:self.mx]
        if w.numel()<self.mn: w=torch.nn.functional.pad(w,(0,self.mn-w.numel()))
        return w, EVENT_TO_IDX.get(r.get("event"),0)

def make_collate(p_mix):
    def collate(b):
        ws=[x[0] for x in b]; ls=[x[1] for x in b]; ow=[]; ol=[]
        for i in range(len(ws)):
            if p_mix>0 and len(ws)>1 and random.random()<p_mix:
                j=random.randrange(len(ws)); mw,ml=lungmix(ws[i],ws[j],ls[i],ls[j]); ow.append(mw); ol.append(ml)
            else: ow.append(ws[i]); ol.append(ls[i])
        return ow, torch.tensor(ol,dtype=torch.long)
    return collate

@torch.no_grad()
def evaluate(model,dl):
    model.eval(); P=[]; G=[]
    for w,l in dl: P.append(model(w)["logits"].argmax(-1).cpu()); G.append(l)
    P=torch.cat(P); G=torch.cat(G); se,sp,sc=icbhi_score(P,G)
    pc={c:(((P==G)&(G==c)).sum().item()/max(int((G==c).sum()),1) if int((G==c).sum()) else None) for c in range(4)}
    return {"icbhi":sc,"se":se,"sp":sp,"pc":pc}

def cw(rows):
    import collections; c=collections.Counter(EVENT_TO_IDX.get(r.get("event"),0) for r in rows)
    t=sum(c.values()); w=torch.tensor([t/(4*max(c.get(i,0),1)) for i in range(4)]); return w/w.mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True); ap.add_argument("--val",required=True)
    ap.add_argument("--init_encoder",default=None)
    ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--batch_size",type=int,default=8)
    ap.add_argument("--lr",type=float,default=3e-5); ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--p_mix",type=float,default=0.5); ap.add_argument("--label_smooth",type=float,default=0.1)
    ap.add_argument("--num_workers",type=int,default=0); ap.add_argument("--log_every",type=int,default=100)
    ap.add_argument("--out",default="checkpoints/sft/ast_strong.pt")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("no CUDA")
    tr,va=WaveDS(a.train),WaveDS(a.val)
    lab=np.array(tr.labels()); cnt=np.bincount(lab,minlength=4).astype(float); cnt[cnt==0]=1
    sm=WeightedRandomSampler(torch.as_tensor(1.0/cnt[lab],dtype=torch.double),len(lab),True)
    tdl=DataLoader(tr,batch_size=a.batch_size,sampler=sm,collate_fn=make_collate(a.p_mix),num_workers=a.num_workers,drop_last=True)
    vdl=DataLoader(va,batch_size=a.batch_size,shuffle=False,collate_fn=make_collate(0.0),num_workers=a.num_workers)
    model=PretrainedAnomalyModel(ASTEncoder(freeze=False)).to(a.device)
    if a.init_encoder and Path(a.init_encoder).exists():
        ck=torch.load(a.init_encoder,map_location="cpu"); model.encoder.load_state_dict(ck.get("encoder",ck),strict=False); print("loaded encoder",a.init_encoder,flush=True)
    weight=cw(tr.rows).to(a.device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    nb=len(tdl); total=nb*a.epochs
    def lr_at(step):
        wsteps=a.warmup*nb
        if step<wsteps: return step/max(wsteps,1)
        prog=(step-wsteps)/max(total-wsteps,1); return 0.5*(1+math.cos(math.pi*prog))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); best=-1.0; gstep=0
    print(f"[strong] {nb} batches/epoch | p_mix {a.p_mix} | classes weighted",flush=True)
    for ep in range(1,a.epochs+1):
        model.train(); run=0.0; n=0; t0=time.time()
        for bi,(w,l) in enumerate(tdl,1):
            for g in opt.param_groups: g["lr"]=a.lr*lr_at(gstep)
            l=l.to(a.device); loss=F.cross_entropy(model(w)["logits"],l,weight=weight,label_smoothing=a.label_smooth)
            opt.zero_grad(); loss.backward(); opt.step(); gstep+=1; run+=loss.item()*l.size(0); n+=l.size(0)
            if bi%a.log_every==0: print(f"  ep{ep} {bi}/{nb} loss {run/max(n,1):.3f} lr {opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/bi:.2f}s/b",flush=True)
        m=evaluate(model,vdl); pc=m["pc"]
        print(f"epoch {ep:02d} loss {run/max(n,1):.4f} ICBHI {m['icbhi']:.4f} (Se {m['se']:.3f} Sp {m['sp']:.3f}) NWCB {pc[0]} {pc[1]} {pc[2]} {pc[3]}",flush=True)
        if m["icbhi"]>best: best=m["icbhi"]; torch.save({"model":model.state_dict(),"epoch":ep,"metrics":m},a.out); print("  -> saved best",round(best,4),flush=True)
    print("done best ICBHI",round(best,4),flush=True)

if __name__=="__main__":
    main()
