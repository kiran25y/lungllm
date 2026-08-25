# REPO PATH: src/lungllm/models/bridge/audio_mapper.py
from __future__ import annotations
import torch
import torch.nn as nn


class AudioMapper(nn.Module):
    def __init__(self, in_dim, llm_dim, k=4, num_layers=2, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, k, in_dim) * 0.02)
        self.inp = nn.Linear(in_dim, in_dim)
        layer = nn.TransformerDecoderLayer(in_dim, num_heads, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(in_dim, llm_dim)

    def forward(self, audio_feat):
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)
        B = audio_feat.size(0)
        q = self.query.expand(B, -1, -1)
        return self.out(self.dec(q, self.inp(audio_feat)))