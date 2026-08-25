# REPO PATH: src/lungllm/data/augment.py
"""SpecAugment (torchaudio, lazy import)."""
from __future__ import annotations


class SpecAugment:
    def __init__(self, freq_mask=24, time_mask=48, n_freq=2, n_time=2, p=0.5):
        self.freq_mask = freq_mask; self.time_mask = time_mask
        self.n_freq = n_freq; self.n_time = n_time; self.p = p; self._fm = self._tm = None

    def _build(self):
        import torchaudio
        self._fm = torchaudio.transforms.FrequencyMasking(self.freq_mask)
        self._tm = torchaudio.transforms.TimeMasking(self.time_mask)

    def __call__(self, spec, meta=None):
        import torch
        if torch.rand(()) > self.p: return spec
        if self._fm is None: self._build()
        for _ in range(self.n_freq): spec = self._fm(spec)
        for _ in range(self.n_time): spec = self._tm(spec)
        return spec
