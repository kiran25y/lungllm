# REPO PATH: src/lungllm/models/sed_model.py
"""Standalone frame-level SED model: shared AST encoder + FrameSEDHead.

Shares the AST encoder with LungLLMMoEv2 by loading the SAME aligned encoder
checkpoint, so SED training benefits the shared representation without tangling
frame-SED into the generative LLM forward path. Train it with its own dataloader
(SEDDataset / collate_sed_wav).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pretrained_encoder import ASTEncoder
from .sed_head import FrameSEDHead, masked_bce_loss
from ..data.sed_labels import NUM_CLASSES


class SEDModel(nn.Module):
    def __init__(
        self,
        encoder: ASTEncoder | None = None,
        freeze_encoder: bool = True,
        aligned_encoder_ckpt: str | None = None,
        **head_kwargs,
    ):
        super().__init__()
        self.encoder = encoder or ASTEncoder(freeze=freeze_encoder)
        if aligned_encoder_ckpt:
            sd = torch.load(aligned_encoder_ckpt, map_location="cpu")
            self.encoder.load_state_dict(sd.get("encoder", sd), strict=False)
        self.head = FrameSEDHead(in_dim=self.encoder.embed_dim, num_classes=NUM_CLASSES, **head_kwargs)

    @torch.no_grad()
    def probe_n_frames(self, seconds: float = 10.24) -> int:
        """Actual encoder T for a window — use to set SEDDataset(n_frames=...)."""
        dummy = [torch.zeros(int(self.encoder.sample_rate * seconds))]
        return self.encoder(dummy, return_frames=True)["frame_features"].shape[1]

    def forward(self, waveforms, targets=None, mask=None, frame_mask=None, pos_weight=None):
        ff = self.encoder(waveforms, return_frames=True)["frame_features"]  # (B, T, D)
        logits = self.head(ff)
        out = {"logits": logits}
        if targets is not None:
            # guard against any off-by-one between rasterizer T and encoder T
            T = min(logits.shape[1], targets.shape[1])
            fm = None if frame_mask is None else frame_mask[:, :T]
            out["loss"] = masked_bce_loss(
                logits[:, :T], targets[:, :T], mask[:, :T], fm, pos_weight=pos_weight
            )
        return out