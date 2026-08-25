# REPO PATH: src/lungllm/data/splits_v2.py
"""Phase 0 — split builders: patient-aware, official ICBHI, OOD (source-disjoint),
and unseen-disease. All read the unified manifest and write parquet/csv splits."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd


def _bucket(pid, seed):
    h = hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _read(p):
    p = Path(p)
    if not p.exists() and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _write(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False); return path
    except Exception:
        path = path.with_suffix(".csv"); df.to_csv(path, index=False); return path


def patient_aware(df, ratios=(0.7, 0.1, 0.2), seed=1337):
    tr, va = ratios[0], ratios[1]
    out = {"train": [], "val": [], "test": []}
    for pid in df["patient_id"].unique():
        b = _bucket(pid, seed)
        k = "train" if b < tr else ("val" if b < tr + va else "test")
        out[k].append(pid)
    return {k: df[df["patient_id"].isin(v)].reset_index(drop=True) for k, v in out.items()}


def ood_split(df, train_datasets, test_datasets):
    """Source-disjoint: train on some datasets, test on others (held-out corpora)."""
    return {"train": df[df["dataset"].isin(train_datasets)].reset_index(drop=True),
            "test": df[df["dataset"].isin(test_datasets)].reset_index(drop=True)}


def unseen_disease_split(df, held_out_diseases):
    """Train on all diseases EXCEPT held_out; test = held-out disease rows (zero-shot)."""
    has = df["diagnosis"].notna()
    held = df["diagnosis"].isin(held_out_diseases)
    return {"train": df[has & ~held].reset_index(drop=True),
            "test": df[has & held].reset_index(drop=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/manifests/all.parquet")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--mode", choices=["patient", "ood", "unseen_disease"], default="patient")
    ap.add_argument("--train_datasets", nargs="+", default=["icbhi", "hf_lung", "sprsound"])
    ap.add_argument("--test_datasets", nargs="+", default=["mendeley"])
    ap.add_argument("--held_out_diseases", nargs="+", default=["Asthma", "Pneumonia"])
    a = ap.parse_args()
    df = _read(a.manifest)
    out = Path(a.out_dir)
    if a.mode == "patient":
        sp = patient_aware(df)
        for k, v in sp.items():
            print(k, len(v), "->", _write(v, out / f"{k}.parquet"))
    elif a.mode == "ood":
        sp = ood_split(df, a.train_datasets, a.test_datasets)
        for k, v in sp.items():
            print("ood", k, len(v), "->", _write(v, out / f"ood_{k}.parquet"))
    else:
        sp = unseen_disease_split(df, a.held_out_diseases)
        for k, v in sp.items():
            print("unseen_disease", k, len(v), "->", _write(v, out / f"unseen_{k}.parquet"))


if __name__ == "__main__":
    main()
