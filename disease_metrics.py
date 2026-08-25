import torch, numpy as np
from torch.utils.data import DataLoader
import lungllm.training.train_multitask as T
from lungllm.models.pretrained_encoder import ASTEncoder, PretrainedMultiTaskModel
try:
    from lungllm.models.moe_model import PretrainedMoEModel
except Exception:
    PretrainedMoEModel = None

def hand_metrics(G, P):
    allc = sorted(set(G.tolist()) | set(P.tolist()))
    per = {}
    for c in allc:
        tp = int(((P==c)&(G==c)).sum()); fp = int(((P==c)&(G!=c)).sum()); fn = int(((P!=c)&(G==c)).sum())
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        per[c] = (prec, rec, f1, int((G==c).sum()))
    tc = [c for c in allc if (G==c).sum() > 0]
    macro_f1 = float(np.mean([per[c][2] for c in tc]))
    bacc     = float(np.mean([per[c][1] for c in tc]))
    return per, macro_f1, bacc

def run(ckpt, moe=False):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    dv = ck.get("disease_vocab", {})
    inv = {v:k for k,v in dv.items()} if isinstance(dv, dict) else {i:str(i) for i in range(len(dv))}
    if moe and PretrainedMoEModel is not None:
        model = PretrainedMoEModel(ASTEncoder(freeze=False), num_anomaly=4, num_disease=len(dv))
    else:
        model = PretrainedMultiTaskModel(ASTEncoder(freeze=False), num_anomaly=4, num_disease=len(dv))
    model.load_state_dict(ck["model"]); model.to("cuda").eval()
    for sp in ["data/splits/test.parquet", "data/splits/ood_test.parquet"]:
        dl = DataLoader(T.DS(sp, dv), batch_size=8, shuffle=False, collate_fn=T.collate)
        P, G = [], []
        with torch.no_grad():
            for w, al, dlab in dl:
                o = model(w)
                if "disease" in o:
                    m = dlab != T.IGNORE
                    if m.any():
                        P.append(o["disease"].argmax(-1).cpu()[m]); G.append(dlab[m])
        if not P:
            print(f"\n== {ckpt} | {sp} == no disease labels"); continue
        P = torch.cat(P).numpy(); G = torch.cat(G).numpy()
        acc = float((P==G).mean())
        _, cnts = np.unique(G, return_counts=True); base = cnts.max()/cnts.sum()
        per, mf1, bacc = hand_metrics(G, P)
        print(f"\n== {ckpt} | {sp} ==")
        print(f"n={len(G)}  acc={acc:.3f}  majority_base_rate={base:.3f}  MACRO_F1={mf1:.3f}  BALANCED_ACC={bacc:.3f}")
        for c in sorted(set(G.tolist())):
            prec, rec, f1, n = per[c]
            print(f"   {str(inv.get(int(c),c))[:26]:26s} n={n:4d}  recall={rec:.3f}  f1={f1:.3f}")

run("checkpoints/sft/ast_multitask.pt", moe=False)
run("checkpoints/sft/ast_moe.pt", moe=True)
