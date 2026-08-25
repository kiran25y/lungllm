# REPO PATH: src/lungllm/models/alignment/losses.py
"""Phase 1 — alignment objectives.

- cka_loss: 1 - linear Centered Kernel Alignment between audio & text batch features
  (rotation/scale invariant; AcuLa). Lower = better aligned.
- info_nce_loss: symmetric CLIP/InfoNCE contrastive loss on L2-normed embeddings
  (RespiraMFM).
- masked_acoustic_loss: MSE reconstruction of masked encoder features (SSM regularizer
  that prevents representation collapse).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _center(gram):
    n = gram.size(0)
    unit = torch.ones(n, n, device=gram.device, dtype=gram.dtype)
    identity = torch.eye(n, device=gram.device, dtype=gram.dtype)
    H = identity - unit / n
    return H @ gram @ H


def linear_cka(X, Y):
    """X: [N, Da], Y: [N, Db]. Returns CKA similarity in [0,1]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    Kx = X @ X.t()
    Ky = Y @ Y.t()
    Kxc, Kyc = _center(Kx), _center(Ky)
    hsic = (Kxc * Kyc).sum()
    denom = (Kxc * Kxc).sum().sqrt() * (Kyc * Kyc).sum().sqrt() + 1e-8
    return hsic / denom


def cka_loss(audio_feat, text_feat):
    return 1.0 - linear_cka(audio_feat, text_feat)


def info_nce_loss(audio_feat, text_feat, temperature=0.07):
    a = F.normalize(audio_feat, dim=-1)
    t = F.normalize(text_feat, dim=-1)
    logits = a @ t.t() / temperature                       # [N, N]
    targets = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def masked_acoustic_loss(pred, target, mask):
    """MSE over masked positions only. mask: 1 where masked/predicted."""
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * mask).sum() / denom
