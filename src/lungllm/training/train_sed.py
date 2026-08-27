# REPO PATH: src/lungllm/training/train_sed.py
"""Train the frame-level SED detector (shares the AST encoder with LungLLMMoEv2).

Pipeline: manifest -> SED records (hf_lung + sprsound) -> patient-disjoint
train/val split -> SEDDataset -> SEDModel (frozen AST + FrameSEDHead) trained
with masked-BCE (+ per-class pos_weight). Reports per-class frame-level Average
Precision and F1 on the validation split. Saves a checkpoint whose 'encoder'
state_dict can be loaded by LungLLMMoEv2(aligned_encoder_ckpt=...).

Run:
  python -m src.lungllm.training.train_sed \
      --manifest data/processed/manifest.parquet \
      --out checkpoints/sed.pt --epochs 15 --batch-size 8
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.sed_records import build_sed_records
from ..data.sed_dataset import SEDDataset, collate_sed_wav, compute_pos_weight
from ..data.sed_labels import CLASSES
from ..models.sed_model import SEDModel


# --------------------------------------------------------------------------- #
# split + metrics
# --------------------------------------------------------------------------- #
def patient_disjoint_split(records, val_frac=0.2, seed=0):
    """Patient-disjoint AND source-stratified: each source contributes val_frac of
    its patients to val, so both HF_Lung and SPRSound appear on both sides."""
    rng = np.random.default_rng(seed)
    by_src = defaultdict(lambda: defaultdict(list))  # source -> patient -> rows
    for r in records:
        by_src[r.get("source", "?")][r.get("patient_id") or r["audio_path"]].append(r)
    train, val = [], []
    for src, by_pat in by_src.items():
        pats = sorted(by_pat)
        rng.shuffle(pats)
        n_val = max(1, int(round(len(pats) * val_frac))) if len(pats) > 1 else 0
        val_pats = set(pats[:n_val])
        train += [r for p in pats if p not in val_pats for r in by_pat[p]]
        val += [r for p in val_pats for r in by_pat[p]]
    return train, val


def _ap(scores, labels):
    """Average precision for one class from scores + binary labels (no sklearn)."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    recall = tp / labels.sum()
    precision = tp / np.clip(tp + fp, 1, None)
    # step-wise AP: sum precision at each positive
    return float((precision * labels).sum() / labels.sum())


def _f1_at(scores, labels, thresh=0.5):
    pred = (scores >= thresh).astype(np.int8)
    tp = int((pred & labels).sum()); fp = int((pred & (1 - labels)).sum())
    fn = int(((1 - pred) & labels).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    per_class = {c: {"s": [], "y": [], "m": []} for c in CLASSES}
    for b in loader:
        out = model([w.to(device) for w in b["waveforms"]])
        probs = torch.sigmoid(out["logits"]).cpu().numpy()      # (B,T,C)
        y = b["targets"].numpy(); m = (b["mask"] * b["frame_mask"].unsqueeze(-1)).numpy()
        T = min(probs.shape[1], y.shape[1])
        for ci, c in enumerate(CLASSES):
            mm = m[:, :T, ci].reshape(-1).astype(bool)
            per_class[c]["s"].append(probs[:, :T, ci].reshape(-1)[mm])
            per_class[c]["y"].append(y[:, :T, ci].reshape(-1)[mm])
    rows = {}
    for c in CLASSES:
        s = np.concatenate(per_class[c]["s"]) if per_class[c]["s"] else np.array([])
        y = np.concatenate(per_class[c]["y"]).astype(np.int8) if per_class[c]["y"] else np.array([])
        rows[c] = (_ap(s, y) if s.size else float("nan"),
                   _f1_at(s, y) if s.size else float("nan"),
                   int(y.sum()) if y.size else 0)
    return rows


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="checkpoints/sed.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--finetune-encoder", action="store_true")
    ap.add_argument("--aligned-encoder-ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap records (smoke test)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers; keep 0 in containers (>0 can deadlock)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_sed] device={device}")
    records = build_sed_records(args.manifest)
    if args.limit:
        records = records[: args.limit]
    train_recs, val_recs = patient_disjoint_split(records, args.val_frac)
    print(f"[train_sed] {len(train_recs)} train / {len(val_recs)} val records")

    model = SEDModel(freeze_encoder=not args.finetune_encoder,
                     aligned_encoder_ckpt=args.aligned_encoder_ckpt).to(device)
    T = model.probe_n_frames()
    train_ds = SEDDataset(train_recs, n_frames=T)
    val_ds = SEDDataset(val_recs, n_frames=T)
    print(f"[train_sed] encoder T={T}; {len(train_ds)} train / {len(val_ds)} val windows")

    pos_weight = compute_pos_weight(train_ds).to(device)
    print("[train_sed] pos_weight:", [round(x, 2) for x in pos_weight.tolist()])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_sed_wav, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_sed_wav, num_workers=args.num_workers)
    n_batches = max(len(train_dl), 1)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for bi, b in enumerate(train_dl):
            out = model([w.to(device) for w in b["waveforms"]],
                        targets=b["targets"].to(device), mask=b["mask"].to(device),
                        frame_mask=b["frame_mask"].to(device), pos_weight=pos_weight)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            tot += out["loss"].item()
            print(f"\r[ep {ep:02d}] batch {bi+1}/{n_batches} loss={out['loss'].item():.4f}",
                  end="", flush=True)
        print()
        sched.step()
        rows = evaluate(model, val_dl, device)
        adv = [c for c in CLASSES if c in ("Wheeze", "Stridor", "Rhonchi", "Crackle")]
        adv_aps = [rows[c][0] for c in adv if not np.isnan(rows[c][0])]
        mAP = float(np.mean(adv_aps)) if adv_aps else float("nan")
        print(f"[ep {ep:02d}] loss={tot/max(len(train_dl),1):.4f} adv_mAP={mAP:.3f} | "
              + " ".join(f"{c}:AP={rows[c][0]:.2f}/F1={rows[c][1]:.2f}(n={rows[c][2]})" for c in CLASSES))
        if mAP > best:
            best = mAP
            import os
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"encoder": model.encoder.state_dict(),
                        "head": model.head.state_dict(),
                        "classes": CLASSES, "n_frames": T, "adv_mAP": best}, args.out)
            print(f"         saved {args.out} (adv_mAP={best:.3f})")
    print(f"[train_sed] done. best adv_mAP={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()