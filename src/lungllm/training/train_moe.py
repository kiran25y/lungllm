# REPO PATH: src/lungllm/training/train_moe.py
"""Train the interpretable MoE model + produce the Expert-Activation Map.

Reuses the multitask data pipeline (DS/collate/canon). Adds the MoE load-balance aux
loss. After training, logs which experts fire for each anomaly class and renders the
Expert-Activation Map figure.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from lungllm.training.train_multitask import DS, collate, canon, read_rows, IGNORE
from lungllm.data.dataset import EVENT_TO_IDX
from lungllm.models.pretrained_encoder import ASTEncoder
from lungllm.models.moe_model import PretrainedMoEModel
from lungllm.training.engine import icbhi_score
from lungllm.models.moe.sparse_moe import EXPERT_NAMES

ANOM = ["normal", "wheeze", "crackle", "both"]


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); ap = []; ag = []; gates_by_class = {c: [] for c in range(4)}
    for w, al, _ in dl:
        o = model(w); pred = o["anomaly"].argmax(-1).cpu()
        ap.append(pred); ag.append(al)
        g = o["moe_info"]["gates"].cpu().numpy()          # [B, num_experts]
        for i, lab in enumerate(al.tolist()):
            if 0 <= lab < 4: gates_by_class[lab].append(g[i])
    ap = torch.cat(ap); ag = torch.cat(ag); se, sp, sc = icbhi_score(ap, ag)
    # mean expert usage per anomaly class -> the Expert-Activation Map matrix
    eam = np.zeros((4, len(EXPERT_NAMES)))
    for c in range(4):
        if gates_by_class[c]:
            eam[c] = np.mean(gates_by_class[c], axis=0)
    return {"icbhi": sc, "se": se, "sp": sp, "eam": eam}


def render_eam(eam, out_path="outputs/figures/expert_activation_map.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4)); plt.imshow(eam, aspect="auto", cmap="viridis")
    plt.xticks(range(len(EXPERT_NAMES)), EXPERT_NAMES, rotation=30, ha="right")
    plt.yticks(range(4), ANOM); plt.xlabel("expert"); plt.ylabel("true anomaly class")
    plt.colorbar(label="mean gate weight"); plt.title("Expert-Activation Map")
    for i in range(4):
        for j in range(len(EXPERT_NAMES)):
            plt.text(j, i, f"{eam[i,j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(); return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train.parquet")
    ap.add_argument("--val", default="data/splits/val.parquet")
    ap.add_argument("--init_encoder", default=None, help="aligned encoder ckpt")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--disease_weight", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=2); ap.add_argument("--num_experts", type=int, default=5)
    ap.add_argument("--finetune", action="store_true"); ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--num_workers", type=int, default=0); ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/sft/ast_moe.pt")
    ap.add_argument("--eam_out", default="outputs/figures/expert_activation_map.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    dv = {d: i for i, d in enumerate(sorted({canon(r.get("diagnosis")) for r in read_rows(a.train)} - {None}))}
    print("[moe] disease classes", len(dv), flush=True)
    tr, va = DS(a.train, dv), DS(a.val, dv)
    if a.balanced_sampler:
        lab = np.array(tr.alabels()); cnt = np.bincount(lab, minlength=4).astype(float); cnt[cnt == 0] = 1
        sm = WeightedRandomSampler(torch.as_tensor(1.0 / cnt[lab], dtype=torch.double), len(lab), True)
        tdl = DataLoader(tr, batch_size=a.batch_size, sampler=sm, collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    else:
        tdl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=a.num_workers, drop_last=True)
    vdl = DataLoader(va, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=a.num_workers)

    enc = ASTEncoder(a.model_name, freeze=not a.finetune)
    if a.init_encoder:
        ck = torch.load(a.init_encoder, map_location="cpu"); enc.load_state_dict(ck.get("encoder", ck), strict=False)
        print("[moe] loaded aligned encoder from", a.init_encoder, flush=True)
    model = PretrainedMoEModel(enc, num_experts=a.num_experts, top_k=a.top_k,
                               num_anomaly=4, num_disease=len(dv)).to(a.device)
    params = [p for p in model.parameters() if p.requires_grad]; nb = len(tdl)
    print(f"[moe] trainable {sum(p.numel() for p in params):,} | {nb} batches/epoch", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); best = -1.0; best_eam = None
    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0; n = 0; t0 = time.time()
        for bi, (w, al, dlab) in enumerate(tdl, 1):
            al = al.to(a.device); dlab = dlab.to(a.device); o = model(w)
            loss = F.cross_entropy(o["anomaly"], al) + o["aux_loss"]
            if (dlab != IGNORE).any(): loss = loss + a.disease_weight * F.cross_entropy(o["disease"], dlab, ignore_index=IGNORE)
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item() * al.size(0); n += al.size(0)
            if bi % a.log_every == 0: print(f"  ep{ep} {bi}/{nb} loss {run/max(n,1):.3f} {(time.time()-t0)/bi:.2f}s/b", flush=True)
        m = evaluate(model, vdl, a.device)
        print(f"epoch {ep:02d} loss {run/max(n,1):.4f} ICBHI {m['icbhi']:.4f} (Se {m['se']:.3f} Sp {m['sp']:.3f})", flush=True)
        if m["icbhi"] > best:
            best = m["icbhi"]; best_eam = m["eam"]
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": {"icbhi": m["icbhi"]},
                        "disease_vocab": dv, "eam": m["eam"]}, a.out)
            print("  -> saved best", round(best, 4), flush=True)
    if best_eam is not None:
        path = render_eam(best_eam, a.eam_out)
        print("[moe] Expert-Activation Map ->", path, flush=True)
        print("[moe] mean expert usage per class (rows=N/W/C/B, cols=%s):" % ",".join(EXPERT_NAMES), flush=True)
        print(np.round(best_eam, 3), flush=True)
    print("done best val ICBHI", round(best, 4), flush=True)


if __name__ == "__main__":
    main()
