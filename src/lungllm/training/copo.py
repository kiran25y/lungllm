# REPO PATH: src/lungllm/training/copo.py
"""Phase 5 — CoPO: acoustic-grounded preference optimization (DPO-style).

Preference pairs are built from ACOUSTIC correctness, not text similarity (this is the
fix for StethoLM's failed BERTScore-mDPO): 'chosen' = a report whose acoustic claims match
the detected event; 'rejected' = a report describing a different/absent finding.
Then run DPO (trl). This builds the preference dataset; plug into trl.DPOTrainer.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

EVENT_TEXT = {"normal": "Auscultation reveals no adventitious sounds.",
              "wheeze": "Auscultation reveals wheezing.",
              "crackle": "Auscultation reveals crackles.",
              "both": "Auscultation reveals crackles and wheezing."}
ALL = list(EVENT_TEXT.values())


def build_pairs(reports_jsonl, out_jsonl):
    """chosen = grounded report; rejected = a mismatched-finding report."""
    rows = [json.loads(l) for l in Path(reports_jsonl).read_text().splitlines() if l.strip()]
    out = Path(out_jsonl); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for r in rows:
            ev = str(r.get("event"))
            chosen = EVENT_TEXT.get(ev)
            if chosen is None:
                continue
            rejected = next((t for t in ALL if t != chosen), ALL[0])
            f.write(json.dumps({"prompt": "Describe the auscultation finding.",
                                "chosen": chosen, "rejected": rejected}) + "\n")
            n += 1
    print(f"wrote {n} acoustic-grounded preference pairs ->", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data/rag/reports/reports.jsonl")
    ap.add_argument("--out", default="data/rag/reports/pref_pairs.jsonl")
    a = ap.parse_args()
    build_pairs(a.reports, a.out)
    print("Next: load pairs into trl.DPOTrainer with your SFT model as the policy.")


if __name__ == "__main__":
    main()
