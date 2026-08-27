# REPO PATH: src/lungllm/data/sed_dataset.py
"""SED dataset for the fixed AST window (~10.24 s / 101 frames).

AST pads/truncates every input to a fixed window, so long recordings are
CHUNKED into consecutive windows (no events dropped) and short clips get their
padded tail FRAME-MASKED (loss ignores silent frames). Each item yields a raw
waveform segment plus per-frame targets/mask for masked-BCE.

Yields per window:
  waveform   : 1-D float32 tensor (raw; AST's feature_extractor re-mels + pads)
  targets    : (T, C) multi-hot
  mask       : (T, C) per-class source validity (phase masked for SPRSound)
  frame_mask : (T,)   1 = real-audio frame, 0 = padded tail
  meta       : dict
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .sed_labels import (
    Event, NUM_CLASSES,
    parse_hf_lung_label, parse_sprsound_label, rasterize,
)


def _load_events(label_path, source) -> list[Event]:
    src = source.lower()
    if src.startswith("hf"):
        return parse_hf_lung_label(label_path)
    return parse_sprsound_label(label_path)


def _window_events(events, w_start, w_end) -> list[Event]:
    """Clip events to [w_start, w_end) and shift to window-local time."""
    out = []
    for e in events:
        s = max(e.start_s, w_start)
        en = min(e.end_s, w_end)
        if en > s:
            out.append(Event(s - w_start, en - w_start, e.classes))
    return out


class SEDDataset(Dataset):
    def __init__(self, records, sample_rate=16000, window_seconds=10.24, n_frames=101):
        """records: list of dicts with keys audio_path, label_path, source[, duration]."""
        self.records = records
        self.sr = sample_rate
        self.win = float(window_seconds)
        self.T = int(n_frames)
        self.min_real = 0.5  # drop a trailing window with < 0.5 s of real audio
        self.index = []  # (rec_i, w_start, real_seconds_in_window)
        for i, r in enumerate(records):
            dur = r.get("duration") or sf.info(r["audio_path"]).duration
            n_win = max(1, int(np.ceil(dur / self.win)))
            for k in range(n_win):
                ws = k * self.win
                real = min(self.win, dur - ws)
                if k > 0 and real < self.min_real:  # negligible tail -> skip
                    continue
                self.index.append((i, ws, real))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        from .features import load_audio  # repo loader: mono + resample to target SR
        rec_i, ws, real = self.index[idx]
        r = self.records[rec_i]

        wav = load_audio(r["audio_path"], self.sr)  # 1-D float32 at self.sr
        win_len = int(round(self.win * self.sr))
        a = int(round(ws * self.sr))
        seg = np.asarray(wav[a:a + win_len], dtype=np.float32)
        if seg.shape[0] < win_len:  # pad short tail so AST/Kaldi fbank has enough samples
            seg = np.pad(seg, (0, win_len - seg.shape[0]))

        events = _window_events(_load_events(r["label_path"], r["source"]), ws, ws + self.win)
        targets, mask = rasterize(events, self.win, r["source"], self.T)

        stride = self.win / self.T
        centers = (np.arange(self.T) + 0.5) * stride
        frame_mask = (centers < real).astype(np.float32)  # 1 = real audio, 0 = padded tail

        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(seg)),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "frame_mask": torch.from_numpy(frame_mask),
            "meta": {"source": r["source"], "path": r["audio_path"], "w_start": ws},
        }


def collate_sed_wav(batch):
    """AST wants a LIST of 1-D waveforms; targets/mask are fixed (T, C) so they stack."""
    return {
        "waveforms": [b["waveform"] for b in batch],
        "targets": torch.stack([b["targets"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "frame_mask": torch.stack([b["frame_mask"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def compute_pos_weight(dataset, eps=1.0):
    """Per-class pos_weight = neg/pos over valid (frame,class) cells. SED is
    negative-dominated; without this the head collapses to all-zeros."""
    pos = np.zeros(NUM_CLASSES, dtype=np.float64)
    valid = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in range(len(dataset)):
        it = dataset[i]
        m = (it["mask"].numpy() * it["frame_mask"].numpy()[:, None])
        t = it["targets"].numpy()
        pos += (t * m).sum(axis=0)
        valid += m.sum(axis=0)
    neg = np.clip(valid - pos, 0, None)
    return torch.tensor(neg / np.clip(pos, eps, None), dtype=torch.float32)