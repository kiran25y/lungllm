"""Frame-level sound-event-detection head + masked-BCE loss.

Consumes `frame_features` (B, T, D) from the encoder and produces per-frame
multi-label logits over the unified SED class space (Inhalation, Exhalation,
Wheeze, Stridor, Rhonchi, Crackle). Masked BCE combines per-class source
validity and per-frame padding validity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.lungllm.data.sed_labels import CLASSES, NUM_CLASSES
except Exception:
    CLASSES = ["Inhalation", "Exhalation", "Wheeze", "Stridor", "Rhonchi", "Crackle"]
    NUM_CLASSES = len(CLASSES)


class FrameSEDHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_classes=NUM_CLASSES,
                 temporal_kernel=5, dropout=0.1, use_temporal_conv=True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            assert temporal_kernel % 2 == 1, "kernel must be odd for 'same' padding"
            self.tconv = nn.Conv1d(in_dim, in_dim, kernel_size=temporal_kernel,
                                   padding=temporal_kernel // 2, groups=in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frame_features):
        x = self.norm(frame_features)
        if self.use_temporal_conv:
            x = x.transpose(1, 2)
            x = self.tconv(x)
            x = x.transpose(1, 2)
        return self.mlp(x)


def masked_bce_loss(logits, targets, mask, frame_mask=None, pos_weight=None):
    full_mask = mask
    if frame_mask is not None:
        full_mask = mask * frame_mask.unsqueeze(-1)
    per_cell = F.binary_cross_entropy_with_logits(
        logits, targets, weight=None, pos_weight=pos_weight, reduction="none")
    per_cell = per_cell * full_mask
    denom = full_mask.sum().clamp_min(1.0)
    return per_cell.sum() / denom


def collate_sed(batch):
    B = len(batch)
    D = batch[0]["features"].shape[-1]
    C = batch[0]["targets"].shape[-1]
    T_max = max(item["features"].shape[0] for item in batch)
    feats = torch.zeros(B, T_max, D)
    tgts = torch.zeros(B, T_max, C)
    msk = torch.zeros(B, T_max, C)
    fmask = torch.zeros(B, T_max)
    for i, item in enumerate(batch):
        Ti = item["features"].shape[0]
        feats[i, :Ti] = torch.as_tensor(item["features"])
        tgts[i, :Ti] = torch.as_tensor(item["targets"])
        msk[i, :Ti] = torch.as_tensor(item["mask"])
        fmask[i, :Ti] = 1.0
    return {"features": feats, "targets": tgts, "mask": msk, "frame_mask": fmask}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, C = 3, 768, NUM_CLASSES
    items = []
    for T in (149, 91, 120):
        feats = torch.randn(T, D)
        tgts = (torch.rand(T, C) > 0.8).float()
        msk = torch.ones(T, C)
        if T == 91:
            msk[:, 0:2] = 0.0
        items.append({"features": feats, "targets": tgts, "mask": msk})
    batch = collate_sed(items)
    head = FrameSEDHead(in_dim=D, num_classes=C)
    logits = head(batch["features"])
    loss = masked_bce_loss(logits, batch["targets"], batch["mask"], batch["frame_mask"])
    loss.backward()
    print("logits:", tuple(logits.shape), "(B, T_max, C)")
    print("loss:", round(loss.item(), 4))
    valid = (batch["mask"] * batch["frame_mask"].unsqueeze(-1)).sum().item()
    print("valid (frame,class) cells:", int(valid))
    print("params:", sum(p.numel() for p in head.parameters()))
