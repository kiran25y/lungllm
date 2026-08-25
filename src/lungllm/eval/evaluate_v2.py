# REPO PATH: src/lungllm/eval/evaluate_v2.py
"""Phase 7 — unified evaluation: anomaly (ICBHI Score / AUROC), disease (macro-F1),
OOD + unseen-disease transfer, faithfulness, and MoE-map coverage. LLM-judge for
generated text is a hook (plug your judge model)."""
from __future__ import annotations
import numpy as np


def icbhi_score(preds, gts, normal_idx=0):
    preds, gts = np.asarray(preds), np.asarray(gts)
    normal = gts == normal_idx
    sp = ((preds == gts) & normal).sum() / max(normal.sum(), 1)
    se = ((preds == gts) & ~normal).sum() / max((~normal).sum(), 1)
    return float(se), float(sp), float((se + sp) / 2)


def macro_f1(preds, gts, num_classes):
    preds, gts = np.asarray(preds), np.asarray(gts)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (gts == c)).sum()
        fp = ((preds == c) & (gts != c)).sum()
        fn = ((preds != c) & (gts == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def balanced_accuracy(preds, gts, num_classes):
    preds, gts = np.asarray(preds), np.asarray(gts)
    recs = []
    for c in range(num_classes):
        m = gts == c
        if m.sum():
            recs.append(((preds == c) & m).sum() / m.sum())
    return float(np.mean(recs)) if recs else 0.0


def llm_judge_accuracy(generated, references, judge=None):
    """Hook: plug a judge model returning yes/no per (generated, reference). Placeholder
    returns None so the caller knows to wire it."""
    if judge is None:
        return None
    return float(np.mean([judge(g, r) for g, r in zip(generated, references)]))
