# REPO PATH: src/lungllm/data/lungmix.py
"""Phase 6 — Lungmix augmentation (Ge et al. 2024): waveform loudness-mask Mixup with
semantic OR-label interpolation. Preserves sparse crackle/wheeze events (loudness mask)
and never lets 'normal' dilute an abnormal label.

Labels are 4-class one-hot indices: 0=normal,1=wheeze,2=crackle,3=both.
"""
from __future__ import annotations

EVENT_BITS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}   # (crackle, wheeze)
BITS_EVENT = {v: k for k, v in EVENT_BITS.items()}


def or_label(a, b):
    """Semantic OR of two 4-class labels -> normal never dilutes; crackle+wheeze=both."""
    ca, wa = EVENT_BITS[int(a)]; cb, wb = EVENT_BITS[int(b)]
    return BITS_EVENT[(ca | cb, wa | wb)]


def lungmix(wav_i, wav_j, y_i, y_j, alpha=1.0, generator=None):
    """Mix two waveforms with a loudness+random mask. Returns (mixed_wav, mixed_label)."""
    import torch
    # align lengths
    n = min(wav_i.numel(), wav_j.numel())
    xi, xj = wav_i[:n], wav_j[:n]
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    # loudness masks: |x| > mean + 2*std  (per-signal salient regions)
    def loud(x):
        thr = x.abs().mean() + 2 * x.abs().std()
        return (x.abs() > thr).float()
    m_loud = lam * torch.clamp(loud(xi) + loud(xj), max=1.0)          # union of salient parts
    rnd = (torch.rand(n, generator=generator) > 0.5).float()          # fill quiet parts
    mask = torch.clamp(m_loud + rnd * (1 - (m_loud > 0).float()), 0.0, 1.0)
    mixed = mask * xi + (1 - mask) * xj
    return mixed, or_label(y_i, y_j)
