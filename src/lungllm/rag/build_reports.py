# REPO PATH: src/lungllm/rag/build_reports.py
"""Disease-aware clinical report corpus. Report = acoustic finding + location + (for
abnormal clips with a diagnosis) a disease-specific auscultation description. Richer /
free-form text -> the model CAN be unfaithful -> real headroom for faithfulness RL."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EVENT_PHRASE = {"normal": "no adventitious sounds", "wheeze": "wheezing",
                "crackle": "crackles", "both": "crackles and wheezing"}

DISEASE_DESC = {
    "asthma": "wheezing and prolonged expiration",
    "copd": "wheezing and decreased breath sounds",
    "pneumonia": "crackles and decreased breath sounds over the affected lobe",
    "bronchitis": "wheezing and rhonchi",
    "chronic bronchitis": "wheezing and rhonchi",
    "lung fibrosis": "fine, dry, end-inspiratory velcro crackles at the bases",
    "heart failure": "fine moist bibasal crackles",
    "pleural effusion": "decreased breath sounds with possible pleural rub",
    "bronchiectasis": "coarse crackles, often persistent and localized",
    "pneumonia ": "crackles over the affected lobe",
}


def canon_dx(x):
    if x is None: return None
    k = str(x).strip().lower()
    return k if k and k != "nan" else None


def template_report(r):
    ev = str(r.get("event"))
    s = f"Auscultation reveals {EVENT_PHRASE.get(ev, 'an adventitious sound')}"
    loc = str(r.get("location") or "").replace("_", " ")
    if loc: s += f" at the {loc}"
    s += "."
    dx = canon_dx(r.get("diagnosis"))
    if dx and ev != "normal":                       # never attach disease to a normal clip
        desc = DISEASE_DESC.get(dx)
        if desc:
            s += f" The pattern is consistent with {dx}, typically presenting as {desc}."
        else:
            s += f" History is consistent with {r.get('diagnosis')}."
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/rag/reports/reports_train.jsonl")
    a = ap.parse_args()
    p = Path(a.manifest)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({"audio_path": r["audio_path"], "report": template_report(r),
                                "event": r.get("event"), "diagnosis": r.get("diagnosis")}) + "\n")
    print(f"wrote {len(df)} disease-aware reports -> {outp}")


if __name__ == "__main__":
    main()