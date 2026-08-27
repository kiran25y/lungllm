# REPO PATH: src/lungllm/data/sed_records.py
"""Build SED training records from the unified manifest.

The manifest is clip-level (one collapsed `event` per recording, no timestamps).
SED needs the per-event label FILE, so we reconstruct its path from audio_path:
  hf_lung : .../<stem>.wav        -> .../<stem>_label.txt
  sprsound: .../<stem>.wav        -> .../<stem>.json
Only these two sources carry frame-level event annotations; ICBHI/Mendeley are
clip-level only and are skipped for SED.

Each record: {audio_path, label_path, source, patient_id}.
`source` is normalized to what sed_labels / SEDDataset expect ('hf_lung'|'sprsound').
"""
from __future__ import annotations

from pathlib import Path

SED_SOURCES = {"hf_lung", "sprsound"}


def _label_path_for(audio_path: str, dataset: str) -> Path | None:
    p = Path(audio_path)
    if dataset == "hf_lung":
        return p.with_name(p.stem + "_label.txt")
    if dataset == "sprsound":
        return p.with_suffix(".json")
    return None


def build_sed_records(manifest, require_exists: bool = True) -> list[dict]:
    """manifest: path to parquet/csv, a DataFrame, or a list of row dicts.
    Returns SED records for hf_lung + sprsound rows with an existing label file."""
    rows = _load_rows(manifest)
    records, missing = [], 0
    seen = set()
    for r in rows:
        ds = str(r.get("dataset", "")).lower()
        if ds not in SED_SOURCES:
            continue
        ap = r.get("audio_path")
        if not ap or ap in seen:  # manifest may repeat a recording across cycles
            continue
        lp = _label_path_for(ap, ds)
        if lp is None:
            continue
        if require_exists and not (Path(ap).exists() and lp.exists()):
            missing += 1
            continue
        seen.add(ap)
        records.append({
            "audio_path": ap,
            "label_path": str(lp),
            "source": ds,
            "patient_id": r.get("patient_id"),
        })
    if missing:
        print(f"[sed_records] skipped {missing} rows with a missing audio/label file")
    print(f"[sed_records] built {len(records)} SED records "
          f"({sum(x['source'] == 'hf_lung' for x in records)} hf_lung, "
          f"{sum(x['source'] == 'sprsound' for x in records)} sprsound)")
    return records


def _load_rows(manifest):
    if isinstance(manifest, list):
        return manifest
    if hasattr(manifest, "to_dict"):  # DataFrame
        return manifest.to_dict("records")
    p = Path(manifest)
    if not p.exists() and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")
    import pandas as pd
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return df.to_dict("records")


if __name__ == "__main__":  # tiny self-test on the path-derivation logic (no disk)
    mock = [
        {"dataset": "hf_lung", "audio_path": "/d/HF_Lung/steth_A.wav", "patient_id": "hflung_steth_A"},
        {"dataset": "sprsound", "audio_path": "/d/new_sprs/train/123_1.wav", "patient_id": "sprsound_123"},
        {"dataset": "icbhi", "audio_path": "/d/icbhi/101_1b1.wav", "patient_id": "icbhi_101"},
        {"dataset": "hf_lung", "audio_path": "/d/HF_Lung/steth_A.wav", "patient_id": "dup"},  # dup
    ]
    recs = build_sed_records(mock, require_exists=False)
    for r in recs:
        print(r["source"], "->", r["label_path"])
    assert len(recs) == 2, "expected 2 unique SED records (icbhi skipped, dup removed)"
    assert recs[0]["label_path"].endswith("steth_A_label.txt")
    assert recs[1]["label_path"].endswith("123_1.json")
    print("OK")