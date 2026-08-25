# REPO PATH: src/lungllm/training/engine.py
"""Collate, train/eval loops, ICBHI metric, class weights, AMP."""
from __future__ import annotations
import contextlib
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def collate_batch(items):
    specs = torch.stack([it["spectrogram"] for it in items])
    labels = torch.tensor([it["anomaly"] for it in items], dtype=torch.long)
    return specs, labels, [it["meta"] for it in items]


def _amp(device, amp):
    if amp and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def icbhi_score(preds, gts, normal_idx=0):
    normal = gts == normal_idx; abn = ~normal
    sp = ((preds == gts) & normal).sum().item() / max(int(normal.sum()), 1)
    se = ((preds == gts) & abn).sum().item() / max(int(abn.sum()), 1)
    return se, sp, (se + sp) / 2


def train_one_epoch(model, loader, opt, device, grad_clip=1.0, amp=False, class_weight=None):
    model.train()
    if class_weight is not None: class_weight = class_weight.to(device)
    run, seen = 0.0, 0
    for specs, labels, _ in loader:
        specs, labels = specs.to(device), labels.to(device)
        with _amp(device, amp):
            out = model(specs)
            loss = F.cross_entropy(out["logits"], labels, weight=class_weight, ignore_index=IGNORE_INDEX) + out["aux_loss"]
        opt.zero_grad(); loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); run += loss.item() * labels.size(0); seen += labels.size(0)
    return run / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval(); preds, gts = [], []
    for specs, labels, _ in loader:
        with _amp(device, amp): out = model(specs.to(device))
        preds.append(out["logits"].argmax(-1).cpu()); gts.append(labels)
    preds = torch.cat(preds); gts = torch.cat(gts)
    se, sp, sc = icbhi_score(preds, gts)
    pc = {c: (((preds == gts) & (gts == c)).sum().item() / max(int((gts == c).sum()), 1)
              if int((gts == c).sum()) else None) for c in range(4)}
    return {"acc": (preds == gts).float().mean().item(), "sensitivity": se, "specificity": sp,
            "icbhi_score": sc, "per_class_recall": pc}


def compute_class_weights(manifest_path, num_classes=4, smoothing=1.0):
    import pandas as pd
    from pathlib import Path
    from lungllm.data.dataset import EVENT_TO_IDX
    p = Path(manifest_path)
    if not p.exists() and p.with_suffix(".csv").exists(): p = p.with_suffix(".csv")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    counts = torch.full((num_classes,), smoothing)
    for ev, c in df["event"].value_counts().items():
        if ev in EVENT_TO_IDX: counts[EVENT_TO_IDX[ev]] += float(c)
    w = counts.sum() / (num_classes * counts); return w / w.mean()
