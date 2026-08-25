# REPO PATH: src/lungllm/data/dataset.py
"""Cycle-level dataset yielding spectrogram + anomaly/severity + meta."""
from __future__ import annotations
from pathlib import Path

EVENT_TO_IDX = {"normal": 0, "wheeze": 1, "crackle": 2, "both": 3}
IGNORE_INDEX = -100


def _base():
    try:
        from torch.utils.data import Dataset
        return Dataset
    except Exception:
        return object


class RespiratoryDataset(_base()):
    def __init__(self, manifest_path, feature_extractor, augment=None):
        import pandas as pd
        p = Path(manifest_path)
        if not p.exists() and p.with_suffix(".csv").exists(): p = p.with_suffix(".csv")
        self.rows = (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)).to_dict("records")
        self.fx = feature_extractor; self.augment = augment

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        from .features import load_audio
        r = self.rows[idx]; wav = load_audio(r["audio_path"], self.fx.sample_rate)
        s = int(float(r["start"]) * self.fx.sample_rate); e = int(float(r["end"]) * self.fx.sample_rate)
        if e > s: wav = wav[s:e]
        spec = self.fx(wav)
        if self.augment is not None: spec = self.augment(spec, meta=r)
        return {"spectrogram": spec, "anomaly": EVENT_TO_IDX.get(r.get("event"), IGNORE_INDEX),
                "meta": {k: r.get(k) for k in ("dataset", "patient_id", "location", "filter_type")}}
