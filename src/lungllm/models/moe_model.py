# REPO PATH: src/lungllm/models/moe_model.py
"""Interpretable MoE model: (aligned) AST encoder -> Sparse MoE (5 named experts)
-> anomaly + disease heads. Logs per-sample gate weights for the Expert-Activation Map."""
from __future__ import annotations
import torch.nn as nn
from .pretrained_encoder import ASTEncoder
from .moe.sparse_moe import SparseMoE, EXPERT_NAMES


class PretrainedMoEModel(nn.Module):
    def __init__(self, encoder, num_experts=5, top_k=2, moe_hidden=1024,
                 num_anomaly=4, num_disease=0, hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        d = encoder.embed_dim
        self.moe = SparseMoE(dim=d, hidden=moe_hidden, num_experts=num_experts, top_k=top_k)
        self.shared = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.anomaly = nn.Linear(hidden, num_anomaly)
        self.disease = nn.Linear(hidden, num_disease) if num_disease > 0 else None
        self.expert_names = EXPERT_NAMES

    def forward(self, waveforms):
        z = self.encoder(waveforms)["clip_embedding"]
        z2, aux, info = self.moe(z)
        h = self.shared(z2)
        out = {"anomaly": self.anomaly(h), "aux_loss": aux, "moe_info": info}
        if self.disease is not None:
            out["disease"] = self.disease(h)
        return out
