"""Remove the per-patient disease label from NORMAL cycles."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = Path(a.manifest)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    before = int(df["diagnosis"].notna().sum())
    df.loc[df["event"] == "normal", "diagnosis"] = None
    after = int(df["diagnosis"].notna().sum())
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    (df.to_parquet(op, index=False) if op.suffix == ".parquet" else df.to_csv(op, index=False))
    print(f"disease labels: {before} -> {after} (removed from {before-after} normal cycles) -> {op}")


if __name__ == "__main__":
    main()
