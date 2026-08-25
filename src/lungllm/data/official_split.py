# REPO PATH: src/lungllm/data/official_split.py
"""Apply the official ICBHI 2017 train/test split (needs ICBHI_challenge_train_test.txt)."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def load_split_file(path):
    m = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line: continue
        p = line.replace("\t", " ").split()
        if len(p) < 2: continue
        stem = p[0][:-4] if p[0].lower().endswith(".wav") else p[0]
        m[stem] = p[1].lower()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/manifests/all.parquet")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    a = ap.parse_args()
    mp = Path(a.manifest)
    if not mp.exists() and mp.with_suffix(".csv").exists(): mp = mp.with_suffix(".csv")
    df = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_csv(mp)
    df = df[df["dataset"] == "icbhi"].copy()
    sm = load_split_file(a.split_file)
    df["stem"] = df["audio_path"].map(lambda p: Path(p).stem)
    df["split"] = df["stem"].map(sm)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for n in ("train", "test"):
        part = df[df["split"] == n].drop(columns=["stem", "split"])
        dest = out / f"icbhi_official_{n}.parquet"
        try: part.to_parquet(dest, index=False)
        except Exception: dest = out / f"icbhi_official_{n}.csv"; part.to_csv(dest, index=False)
        print(f"[official] {n}: {len(part)} cycles -> {dest}")


if __name__ == "__main__":
    main()
