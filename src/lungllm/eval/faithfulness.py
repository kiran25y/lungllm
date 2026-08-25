# REPO PATH: src/lungllm/eval/faithfulness.py
"""Phase 5 — Acoustic FActScore: fraction of a report's acoustic claims that are
supported by the detected events. Simple keyword grounding (extend with an NLI model)."""
from __future__ import annotations

CLAIM_KEYWORDS = {"wheeze": ["wheez"], "crackle": ["crackl", "crepit", "rale"],
                  "normal": ["no adventitious", "normal", "clear"]}


def detected_set(event):
    e = str(event)
    if e == "both": return {"wheeze", "crackle"}
    if e in ("wheeze", "crackle"): return {e}
    return {"normal"}


def acoustic_factscore(report, event):
    """Return precision = supported_claims / total_claims for one report."""
    text = report.lower()
    detected = detected_set(event)
    claims, supported = 0, 0
    for label, kws in CLAIM_KEYWORDS.items():
        if any(k in text for k in kws):
            claims += 1
            if label in detected:
                supported += 1
    return supported / claims if claims else 1.0   # no claim -> vacuously faithful


def corpus_factscore(reports, events):
    scores = [acoustic_factscore(r, e) for r, e in zip(reports, events)]
    return sum(scores) / max(len(scores), 1)
