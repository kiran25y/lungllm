# REPO PATH: src/lungllm/models/alignment/heads.py
"""Phase 1 — projection heads mapping audio and text embeddings into a shared space."""
from __future__ import annotations
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)
