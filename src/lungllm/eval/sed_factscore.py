# REPO PATH: src/lungllm/eval/sed_factscore.py
"""Acoustic FActScore grounded in TYPED, frame-level SED detections.

Upgrade over eval/faithfulness.py: instead of grounding a report's acoustic
claims in a single clip-level label (`detected_set(event)`), we ground them in
what the SED detector actually localized — wheeze / crackle / rhonchi / stridor,
each with frame support and (optionally) the breath phase it overlaps. This is
the acoustic-grounding that CoPO/GRPO/eval reward, and it is strictly richer than
the clip tag: a report claiming "expiratory wheeze" can now be checked against
whether a wheeze was detected AND whether it fell in exhalation.

Backward compatible: if you pass `event=` (clip label) and no `detections=`,
it reproduces the old clip-level behavior.
"""
from __future__ import annotations

import numpy as np

from ..data.sed_labels import ADV_CLASSES, PHASE_CLASSES, CLASS_TO_IDX

# claim vocabulary per canonical adventitious class (+ normal)
CLAIM_KEYWORDS = {
    "wheeze": ["wheez"],
    "crackle": ["crackl", "crepit", "rale"],
    "rhonchi": ["rhonch"],
    "stridor": ["stridor"],
    "normal": ["no adventitious", "no abnormal", "normal breath", "clear", "unremarkable"],
}
PHASE_KEYWORDS = {  # phrases that assert a phase for an adventitious claim
    "Inhalation": ["inspir", "inhal", "on inspiration"],
    "Exhalation": ["expir", "exhal", "on expiration"],
}
ADV_LOWER = [c.lower() for c in ADV_CLASSES]  # ['wheeze','stridor','rhonchi','crackle']


# --------------------------------------------------------------------------- #
# SED logits -> detections
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def frame_preds(logits, frame_mask=None, thresh=0.5):
    """logits (T, C) -> binary preds (T, C), zeroing padded frames."""
    preds = (_sigmoid(logits) >= thresh).astype(np.int8)
    if frame_mask is not None:
        preds = preds * np.asarray(frame_mask, dtype=np.int8)[:, None]
    return preds


def detections(logits, frame_mask=None, thresh=0.5, min_frames=2, with_phase=False):
    """Return the set of typed adventitious detections (lowercased class names).

    If with_phase, return {event: set(phases)} where phase is the breath phase(s)
    the event's active frames overlap (only meaningful when phase channels are
    valid, i.e. HF_Lung). Events with < min_frames support are dropped.
    """
    preds = frame_preds(logits, frame_mask, thresh)
    present = {}
    for cname in ADV_CLASSES:
        col = preds[:, CLASS_TO_IDX[cname]]
        if col.sum() < min_frames:
            continue
        ev = cname.lower()
        if not with_phase:
            present[ev] = set()
            continue
        phases = set()
        active = col.astype(bool)
        for ph in PHASE_CLASSES:
            if (preds[:, CLASS_TO_IDX[ph]].astype(bool) & active).sum() >= 1:
                phases.add(ph)
        present[ev] = phases
    return present if with_phase else set(present.keys())


# --------------------------------------------------------------------------- #
# FActScore
# --------------------------------------------------------------------------- #
def _clip_event_to_set(event):
    """Backward-compat: old clip label -> detection set."""
    e = str(event)
    if e == "both":
        return {"wheeze", "crackle"}
    if e in ("wheeze", "crackle", "rhonchi", "stridor"):
        return {e}
    return set()  # normal / unknown -> no adventitious


def acoustic_factscore(report, detections=None, event=None, check_phase=False):
    """precision = supported_claims / total_claims for one report.

    - detections: a set of typed events (from `detections(...)`) OR, if
      check_phase, a dict {event: set(phases)}. Preferred (SED-grounded).
    - event: fallback clip label if detections is None (old behavior).
    A claim is (a) an adventitious type mentioned, or (b) a 'normal' assertion.
    Supported iff the SED detections agree. Phase claims (inspiratory/expiratory)
    are checked against the detection's overlapping phase when check_phase=True.
    """
    text = report.lower()

    if detections is None:
        det_set, det_phase = _clip_event_to_set(event), {}
    elif isinstance(detections, dict):
        det_set, det_phase = set(detections.keys()), detections
    else:
        det_set, det_phase = set(detections), {}

    claims = supported = 0

    # adventitious-type claims
    for label in ADV_LOWER:
        kws = CLAIM_KEYWORDS[label]
        if any(k in text for k in kws):
            claims += 1
            ok = label in det_set
            if ok and check_phase:
                asserted = {ph for ph, pk in PHASE_KEYWORDS.items() if any(w in text for w in pk)}
                if asserted:  # report pins a phase -> require overlap with detected phase
                    ok = bool(asserted & det_phase.get(label, set()))
            supported += int(ok)

    # 'normal / clear' claim -> supported iff nothing adventitious was detected
    if any(k in text for k in CLAIM_KEYWORDS["normal"]):
        claims += 1
        supported += int(len(det_set) == 0)

    return supported / claims if claims else 1.0  # no acoustic claim -> vacuously faithful


def corpus_factscore(reports, detections_list=None, events=None, check_phase=False):
    """Mean FActScore over a corpus. Provide detections_list (SED) or events (clip)."""
    n = len(reports)
    if detections_list is None:
        detections_list = [None] * n
    if events is None:
        events = [None] * n
    scores = [
        acoustic_factscore(r, d, e, check_phase)
        for r, d, e in zip(reports, detections_list, events)
    ]
    return sum(scores) / max(len(scores), 1)


if __name__ == "__main__":  # pure-python self-test (mock logits + reports)
    T, C = 101, 6
    lo = np.full((T, C), -5.0)  # all-negative baseline
    # detect a wheeze on frames 10..40, overlapping an exhalation on 5..45
    lo[10:40, CLASS_TO_IDX["Wheeze"]] = 5.0
    lo[5:45, CLASS_TO_IDX["Exhalation"]] = 5.0
    det = detections(lo, with_phase=True)
    print("detections+phase:", {k: sorted(v) for k, v in det.items()})

    faithful = "Expiratory wheeze is present."          # correct type + phase
    wrong_phase = "Inspiratory wheeze is heard."         # right type, wrong phase
    hallucinated = "Coarse crackles throughout."         # type not detected
    clean = "Normal breath sounds, no adventitious sounds."

    for rep in (faithful, wrong_phase, hallucinated, clean):
        s_type = acoustic_factscore(rep, detections=set(det.keys()))
        s_phase = acoustic_factscore(rep, detections=det, check_phase=True)
        print(f"  type={s_type:.2f} phase={s_phase:.2f} | {rep}")
    # expected: faithful 1.00/1.00 ; wrong_phase 1.00/0.00 ; hallucinated 0.00/0.00 ; clean 0.00 (claims normal but wheeze present)