# REPO PATH: src/lungllm/models/moe/sparse_moe.py
"""Sparse MoE (5 experts, top-2) with load-balance loss + gate logging for the EAM."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_NAMES = ["wheeze", "crackle", "both", "normal", "severity"]


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
    def forward(self, x): return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self, dim=768, hidden=1024, num_experts=5, top_k=2, router_noise=True,
                 load_balance_weight=0.01, noise_std=1.0):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.dim = dim; self.num_experts = num_experts; self.top_k = top_k
        self.router_noise = router_noise; self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std
        self.experts = nn.ModuleList(Expert(dim, hidden) for _ in range(num_experts))
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x):
        lead = x.shape[:-1]; xf = x.reshape(-1, self.dim)
        logits = self.router(xf)
        if self.router_noise and self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std
        gates = F.softmax(logits, dim=-1)
        val, idx = torch.topk(gates, self.top_k, dim=-1)
        w = val / (val.sum(dim=-1, keepdim=True) + 1e-9)
        gate = torch.zeros_like(gates); gate.scatter_(1, idx, w)
        exp_out = torch.stack([e(xf) for e in self.experts], dim=1)
        out = (gate.unsqueeze(-1) * exp_out).sum(dim=1)
        importance = gates.mean(dim=0)
        routed = torch.zeros_like(gates); routed.scatter_(1, idx, 1.0); load = routed.mean(dim=0)
        aux = self.load_balance_weight * self.num_experts * torch.sum(importance * load)
        info = {"gates": gates.reshape(*lead, self.num_experts).detach(),
                "top_idx": idx.reshape(*lead, self.top_k).detach(), "load": load.detach()}
        return out.reshape(*lead, self.dim), aux, info
