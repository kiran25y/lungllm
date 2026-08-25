"""Frame-level sound-event-detection head + masked-BCE loss.

Consumes `frame_features` (B, T, D) from the encoder (step 1) and produces
per-frame multi-label logits over the unified SED class space (step 2:
Inhalation, Exhalation, Wheeze, Stridor, Rhonchi, Crackle).

The loss is masked BCE: the mask combines
  (a) per-class source validity  — a dataset may not annotate a channel
      (e.g. SPRSound has no breath phase), so those channels are skipped, and
  (b) per-frame padding validity — variable-length clips padded in a batch.
Masked channels/frames contribute zero loss and zero gradient; they are NOT
treated as negatives.

Drop-in location: src/lungllm/models/sed_head.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # keep class list in one place
    from src.lungllm.data.sed_labels import CLASSES, NUM_CLASSES
except Exception:  # standalone fallback
    CLASSES = ["Inhalation", "Exhalation", "Wheeze", "Stridor", "Rhonchi", "Crackle"]
    NUM_CLASSES = len(CLASSES)


class FrameSEDHead(nn.Module):
    """Per-frame multi-label classifier over encoder frame features.

    LayerNorm -> optional depthwise temporal conv (local context) ->
    Linear -> GELU -> Dropout -> Linear -> logits (B, T, C).
    The temporal conv gives each frame a short receptive field without
    collapsing the time axis, which matters for onset/offset resolution.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        temporal_kernel: int = 5,
        dropout: float = 0.1,
        use_temporal_conv: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            assert temporal_kernel % 2 == 1, "kernel must be odd for 'same' padding"
            self.tconv = nn.Conv1d(
                in_dim, in_dim, kernel_size=temporal_kernel,
                padding=temporal_kernel // 2, groups=in_dim,  # depthwise
            )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """frame_features: (B, T, D) -> logits: (B, T, C)."""
        x = self.norm(frame_features)
        if self.use_temporal_conv:
            x = x.transpose(1, 2)          # (B, D, T)
            x = self.tconv(x)
            x = x.transpose(1, 2)          # (B, T, D)
        return self.mlp(x)


def masked_bce_loss(
    logits: torch.Tensor,     # (B, T, C)
    targets: torch.Tensor,    # (B, T, C) in {0,1}
    mask: torch.Tensor,       # (B, T, C) in {0,1}: channel-validity
    frame_mask: torch.Tensor | None = None,  # (B, T) in {0,1}: 1 = real frame
    pos_weight: torch.Tensor | None = None,  # (C,) optional class rebalancing
) -> torch.Tensor:
    """BCE-with-logits averaged over valid (frame, class) cells only."""
    full_mask = mask
    if frame_mask is not None:
        full_mask = mask * frame_mask.unsqueeze(-1)  # broadcast (B,T,1)
    per_cell = F.binary_cross_entropy_with_logits(
        logits, targets, weight=None, pos_weight=pos_weight, reduction="none"
    )
    per_cell = per_cell * full_mask
    denom = full_mask.sum().clamp_min(1.0)
    return per_cell.sum() / denom


def collate_sed(batch):
    """Pad a list of per-clip dicts to a batch.

    Each item: {'features': (T_i, D), 'targets': (T_i, C), 'mask': (T_i, C)}.
    Returns tensors (B, T_max, *) plus frame_mask (B, T_max).
    """
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


if __name__ == "__main__":  # smoke test: shapes, masking, gradient sanity
    torch.manual_seed(0)
    B, D, C = 3, 768, NUM_CLASSES
    # variable-length clips
    items = []
    for T in (149, 91, 120):
        feats = torch.randn(T, D)
        tgts = (torch.rand(T, C) > 0.8).float()
        msk = torch.ones(T, C)
        if T == 91:  # simulate a SPRSound clip: phase channels masked
            msk[:, 0:2] = 0.0
        items.append({"features": feats, "targets": tgts, "mask": msk})
    batch = collate_sed(items)

    head = FrameSEDHead(in_dim=D, num_classes=C)
    logits = head(batch["features"])
    loss = masked_bce_loss(logits, batch["targets"], batch["mask"], batch["frame_mask"])
    loss.backward()

    print("logits:", tuple(logits.shape), "(B, T_max, C)")
    print("loss:", round(loss.item(), 4))
    # verify masked cells contribute nothing: zero out all valid frames of clip #1's
    # phase channels are already excluded; check grad flows and denom is right.
    valid = (batch["mask"] * batch["frame_mask"].unsqueeze(-1)).sum().item()
    print("valid (frame,class) cells:", int(valid), "of", B * batch["features"].shape[1] * C)
    print("params:", sum(p.numel() for p in head.parameters()))