"""Unified sound-event-detection (SED) label rasterizer for HF_Lung + SPRSound.

Normalizes both annotation formats to a common event schema, then rasterizes
events onto the encoder's frame grid. Stride is derived per-clip as
`duration / encoder_T` so alignment is frame-perfect regardless of clip length
(no hardcoded ms/frame).

Label space (multi-label, two groups):
  PHASE:        Inhalation, Exhalation            (HF_Lung only; masked for SPRSound)
  ADVENTITIOUS: Wheeze, Stridor, Rhonchi, Crackle (both datasets)

Masking (masked-BCE): channels a source does NOT annotate are marked invalid in
the returned mask so the loss skips them. They are NOT treated as negatives.

Drop-in location: src/lungllm/data/sed_labels.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Canonical label space
# --------------------------------------------------------------------------- #
PHASE_CLASSES = ["Inhalation", "Exhalation"]
ADV_CLASSES = ["Wheeze", "Stridor", "Rhonchi", "Crackle"]
CLASSES = PHASE_CLASSES + ADV_CLASSES
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# The one semantic assumption worth confirming: HF_Lung "D" tag.
# Per HF_Lung_V1 docs, D = Discontinuous Adventitious Sound = Crackle.
HF_D_MEANS = "Crackle"

# --------------------------------------------------------------------------- #
# Raw-type -> canonical mapping
# --------------------------------------------------------------------------- #
HF_MAP = {
    "I": "Inhalation",
    "E": "Exhalation",
    "D": HF_D_MEANS,
    "Wheeze": "Wheeze",
    "Stridor": "Stridor",
    "Rhonchi": "Rhonchi",
}

# SPRSound event types. A list value = multi-label decomposition.
SPRS_MAP = {
    "Normal": None,  # absence of adventitious; no positive channel
    "Poor Quality": None,
    "Wheeze": "Wheeze",
    "Stridor": "Stridor",
    "Rhonchi": "Rhonchi",
    "Coarse Crackle": "Crackle",
    "Fine Crackle": "Crackle",
    "Wheeze+Crackle": ["Wheeze", "Crackle"],
    "Wheeze&Crackle": ["Wheeze", "Crackle"],
}


@dataclass
class Event:
    start_s: float
    end_s: float
    classes: list  # canonical class names (already mapped/expanded)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def _timecode_to_seconds(tc: str) -> float:
    """'HH:MM:SS.mmm' -> float seconds."""
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_hf_lung_label(path) -> list[Event]:
    """Parse a HF_Lung '*_label.txt' file: 'TYPE HH:MM:SS.mmm HH:MM:SS.mmm' per line."""
    events: list[Event] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        typ, start, end = parts[0], parts[1], parts[2]
        canon = HF_MAP.get(typ)
        if canon is None:
            continue
        events.append(Event(_timecode_to_seconds(start), _timecode_to_seconds(end), [canon]))
    return events


def parse_sprsound_label(path) -> list[Event]:
    """Parse a SPRSound '*.json': {'event_annotation':[{'start','end','type'}]} (ms)."""
    data = json.loads(Path(path).read_text())
    events: list[Event] = []
    for ev in data.get("event_annotation", []):
        start_s = float(ev["start"]) / 1000.0
        end_s = float(ev["end"]) / 1000.0
        raw = ev["type"]
        mapped = SPRS_MAP[raw] if raw in SPRS_MAP else (raw if raw in CLASS_TO_IDX else None)
        if mapped is None:
            continue
        canon = mapped if isinstance(mapped, list) else [mapped]
        events.append(Event(start_s, end_s, canon))
    return events


# --------------------------------------------------------------------------- #
# Masking + rasterization
# --------------------------------------------------------------------------- #
def source_channel_mask(source: str) -> np.ndarray:
    """1 = source annotates this channel (loss applies); 0 = unknown (skip)."""
    m = np.ones(NUM_CLASSES, dtype=np.float32)
    if source.lower() in ("sprsound", "sprs", "new_sprs"):
        for c in PHASE_CLASSES:  # SPRSound has no breath-phase annotations
            m[CLASS_TO_IDX[c]] = 0.0
    return m


def rasterize(
    events: list[Event],
    duration_s: float,
    source: str,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize events onto `num_frames` frames (= encoder output T for this clip).

    Stride is derived as duration_s / num_frames for frame-perfect alignment.
    Returns (targets[T, C] float{0,1}, mask[T, C] float{0,1}).
    """
    T = int(num_frames)
    stride = duration_s / T if T > 0 else duration_s
    targets = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    for ev in events:
        f0 = max(0, min(T, int(np.floor(ev.start_s / stride))))
        f1 = max(0, min(T, int(np.ceil(ev.end_s / stride))))
        for c in ev.classes:
            idx = CLASS_TO_IDX.get(c)
            if idx is not None:
                targets[f0:f1, idx] = 1.0
    ch = source_channel_mask(source)  # [C]
    mask = np.broadcast_to(ch, (T, NUM_CLASSES)).copy()
    return targets, mask


def build_frame_targets(label_path, source: str, duration_s: float, num_frames: int):
    """Convenience: parse + rasterize by source. `num_frames` = encoder T for the clip."""
    src = source.lower()
    if src.startswith("hf"):
        events = parse_hf_lung_label(label_path)
    elif src in ("sprsound", "sprs", "new_sprs"):
        events = parse_sprsound_label(label_path)
    else:
        raise ValueError(f"unknown source: {source!r}")
    return rasterize(events, duration_s, source, num_frames)


if __name__ == "__main__":  # tiny smoke test with synthetic events
    evs = [Event(1.5, 2.457, ["Inhalation"]), Event(12.754, 13.485, ["Rhonchi"])]
    t, m = rasterize(evs, duration_s=15.0, source="hf_lung", num_frames=101)
    print("targets", t.shape, "active frames/class:", t.sum(0).astype(int).tolist())
    print("mask row0 (hf):", m[0].astype(int).tolist(), "classes:", CLASSES)
    _, m2 = rasterize([], duration_s=8.0, source="sprsound", num_frames=80)
    print("mask row0 (sprs, phase masked):", m2[0].astype(int).tolist())