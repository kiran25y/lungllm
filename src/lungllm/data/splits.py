# REPO PATH: src/lungllm/data/splits.py
"""Patient-aware train/val/test split (no patient leakage)."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd


def _bucket(pid, seed):
    return int(hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def patient_aware_split(df, ratios, seed=1337):
    tr, va = ratios["train"], ratios["val"]; assign = {}
    for pid in df["patient_id"].unique():
        b = _bucket(pid, seed)
        assign[pid] = "train" if b < tr else ("val" if b < tr + va else "test")
    col = df["patient_id"].map(assign)
    return {n: df[col == n].reset_index(drop=True) for n in ("train", "val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/manifests/all.parquet")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    mp = Path(a.manifest)
    if not mp.exists() and mp.with_suffix(".csv").exists(): mp = mp.with_suffix(".csv")
    df = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_csv(mp)
    sp = patient_aware_split(df, {"train": a.train, "val": a.val, "test": a.test}, a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for n, part in sp.items():
        dest = out / f"{n}.parquet"
        try: part.to_parquet(dest, index=False)
        except Exception: dest = out / f"{n}.csv"; part.to_csv(dest, index=False)
        print(f"[splits] {n}: {len(part)} rows / {part['patient_id'].nunique()} patients -> {dest}")


if __name__ == "__main__":
    main()
