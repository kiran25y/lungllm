# REPO PATH: src/lungllm/eval/sed_grounding.py
"""Typed acoustic grounding for the RL reward (GRPO/CoPO) and eval.

Produces a typed detection set {wheeze, crackle, rhonchi, stridor} (optionally
with breath phase) from two sources, both feeding sed_factscore.acoustic_factscore:

  * clip_detections_from_label(...)  -> VERIFIABLE ground truth from the annotation
    file. Use for the RL reward on TRAINING clips (labels available). Strictly
    richer than the manifest's clip `event`, which collapses rhonchi/stridor.
  * clip_detections_from_sed(...)    -> the trained detector's prediction. Use for
    eval_faithfulness on held-out clips and at deployment (no labels).

Both return either a set of event names, or {event: set(phases)} when with_phase.
"""
from __future__ import annotations

from ..data.sed_labels import (
    parse_hf_lung_label, parse_sprsound_label, ADV_CLASSES, PHASE_CLASSES,
)

_ADV = {c.lower() for c in ADV_CLASSES}


def _overlap(a, b):
    return max(a.start_s, b.start_s) < min(a.end_s, b.end_s)


def clip_detections_from_label(label_path, source, with_phase=False):
    """Typed detections present anywhere in the clip, from the annotation file."""
    events = (parse_hf_lung_label(label_path) if str(source).lower().startswith("hf")
              else parse_sprsound_label(label_path))
    present: dict[str, set] = {}
    for e in events:
        for c in e.classes:
            if c.lower() in _ADV:
                present.setdefault(c.lower(), set())
    if with_phase:
        phase_events = [(e, [c for c in e.classes if c in PHASE_CLASSES]) for e in events]
        phase_events = [(e, ph[0]) for e, ph in phase_events if ph]
        for e in events:
            adv_here = [c.lower() for c in e.classes if c.lower() in _ADV]
            if not adv_here:
                continue
            for pe, ph in phase_events:
                if _overlap(e, pe):
                    for a in adv_here:
                        present[a].add(ph)
    return present if with_phase else set(present.keys())


def clip_detections_from_sed(model, waveform, sr=16000, window_seconds=10.24,
                             thresh=0.5, min_frames=2, with_phase=False):
    """Typed detections from the trained SED model, aggregated across windows.

    `model`: a SEDModel (encoder + head). `waveform`: 1-D tensor/array at `sr`.
    Chunks like SEDDataset, runs the detector per window, unions detections.
    """
    import numpy as np
    import torch
    from .sed_factscore import detections as _dets  # logits -> typed detections

    wav = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
    win = int(round(window_seconds * sr))
    n_win = max(1, int(np.ceil(len(wav) / win)))
    agg: dict[str, set] = {}
    model.eval()
    with torch.no_grad():
        for k in range(n_win):
            seg = wav[k * win:(k + 1) * win]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            logits = model([torch.from_numpy(seg.astype("float32"))])["logits"][0].cpu().numpy()
            d = _dets(logits, thresh=thresh, min_frames=min_frames, with_phase=with_phase)
            if with_phase:
                for ev, ph in d.items():
                    agg.setdefault(ev, set()).update(ph)
            else:
                for ev in d:
                    agg.setdefault(ev, set())
    return agg if with_phase else set(agg.keys())


if __name__ == "__main__":  # test the label path on a synthetic HF_Lung label
    import tempfile, os
    txt = ("I 00:00:01.000 00:00:02.000\n"      # inhalation
           "Wheeze 00:00:01.200 00:00:01.800\n"  # wheeze during inhalation
           "E 00:00:02.100 00:00:03.000\n"       # exhalation
           "D 00:00:02.200 00:00:02.600\n")      # crackle during exhalation
    f = tempfile.NamedTemporaryFile("w", suffix="_label.txt", delete=False); f.write(txt); f.close()
    print("types:", clip_detections_from_label(f.name, "hf_lung"))
    print("typed+phase:", {k: sorted(v) for k, v in
                           clip_detections_from_label(f.name, "hf_lung", with_phase=True).items()})
    os.unlink(f.name)
    # expected: {'wheeze','crackle'} ; wheeze->[Inhalation], crackle->[Exhalation]