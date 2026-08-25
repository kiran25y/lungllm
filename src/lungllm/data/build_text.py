# REPO PATH: src/lungllm/data/build_text.py
"""Phase 0 — add a `text` column (symptom/demographic/context string) to the manifest,
built from the fields we already parse (diagnosis, location, filter, dataset)."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def row_to_text(r):
    parts = []
    if r.get("diagnosis"): parts.append(f"reported condition: {r['diagnosis']}")
    if r.get("location"): parts.append(f"auscultation site: {str(r['location']).replace('_', ' ')}")
    if r.get("filter_type"): parts.append(f"stethoscope: {r['filter_type']}")
    parts.append(f"source: {r.get('dataset','unknown')}")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = Path(a.manifest)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df["text"] = df.apply(row_to_text, axis=1)
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    (df.to_parquet(op, index=False) if op.suffix == ".parquet" else df.to_csv(op, index=False))
    print(f"wrote {len(df)} rows with text ->", op)
    print(df["text"].head(3).to_string())


if __name__ == "__main__":
    main()
