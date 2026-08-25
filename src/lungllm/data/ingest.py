# REPO PATH: src/lungllm/data/ingest.py
"""Ingest ICBHI + Mendeley + HF Lung + SPRSound into one unified manifest."""
from __future__ import annotations
import argparse, re, json
from pathlib import Path
import pandas as pd

UNIFIED_COLUMNS = ["audio_path","dataset","patient_id","start","end","event","phase",
                   "diagnosis","severity","location","filter_type"]
ICBHI_LOC = {"Tc":"trachea","Al":"anterior_left","Ar":"anterior_right","Pl":"posterior_left",
             "Pr":"posterior_right","Ll":"lateral_left","Lr":"lateral_right"}


def _ev(cr, wh):
    return "both" if cr and wh else ("crackle" if cr else ("wheeze" if wh else "normal"))


def _icbhi_name(stem):
    p = stem.split("_")
    if len(p) < 5: raise ValueError(stem)
    return {"patient_num": p[0], "location": ICBHI_LOC.get(p[2], p[2]), "equipment": p[4]}


def _icbhi_diag(raw):
    for n in ("patient_diagnosis.csv","ICBHI_Challenge_diagnosis.txt"):
        f = raw / n
        if f.exists():
            d = {}
            for line in f.read_text().splitlines():
                line = line.strip().replace("\t", ",")
                if not line: continue
                pid, _, lab = line.partition(","); d[re.sub(r"\D","",pid)] = lab.strip()
            return d
    # also look under raw/label/ or parent
    for cand in list(raw.rglob("*iagnosis*.csv")) + list(raw.rglob("*iagnosis*.txt")):
        d = {}
        for line in cand.read_text().splitlines():
            line = line.strip().replace("\t", ",")
            if not line: continue
            pid, _, lab = line.partition(","); d[re.sub(r"\D","",pid)] = lab.strip()
        if d: return d
    return {}


def parse_icbhi(raw):
    raw = Path(raw); diag = _icbhi_diag(raw); rows = []
    skip = {"patient_diagnosis","ICBHI_Challenge_diagnosis","filename_differences","filename_format"}
    for txt in sorted(p for p in raw.rglob("*.txt") if p.stem not in skip and "iagnosis" not in p.stem):
        try: m = _icbhi_name(txt.stem)
        except ValueError: continue
        wav = txt.with_suffix(".wav"); dx = diag.get(m["patient_num"])
        for line in txt.read_text().splitlines():
            f = line.split()
            if len(f) < 4: continue
            rows.append({"audio_path":str(wav),"dataset":"icbhi","patient_id":f"icbhi_{m['patient_num']}",
                         "start":float(f[0]),"end":float(f[1]),"event":_ev(int(f[2]),int(f[3])),
                         "phase":None,"diagnosis":dx,"severity":None,"location":m["location"],
                         "filter_type":m["equipment"]})
    return rows


_MF = {"B":"Bell","D":"Diaphragm","E":"Extended","P":"Progressive"}
_MD = {"n":"Normal","normal":"Normal","asthma":"Asthma","copd":"COPD","bron":"Bronchiectasis",
       "bronchiectasis":"Bronchiectasis","bronchiolitis":"Bronchiolitis","pneumonia":"Pneumonia",
       "heart failure":"Heart Failure","lung fibrosis":"Lung Fibrosis","fibrosis":"Lung Fibrosis",
       "plueral effusion":"Pleural Effusion","pleural effusion":"Pleural Effusion"}


def _md(x):
    k = x.strip().lower(); return _MD.get(k, x.strip().title() if x.strip() else None)


def _mp(s):
    s = s.strip().lower()
    ph = "expiration" if s.startswith("e") else ("inspiration" if s.startswith("i") else None)
    if s in ("n","normal",""): return "normal", None
    hw = "wheez" in s or re.search(r"\bw\b", s); hc = "crep" in s or "crackl" in s or re.search(r"\bc\b", s)
    if hw and hc: return "both", ph
    if hw: return "wheeze", ph
    if hc: return "crackle", ph
    return "normal", ph


def _ml(x):
    p = x.strip().split(); m1={"P":"posterior","A":"anterior"}; m2={"L":"left","R":"right"}
    m3={"U":"upper","M":"middle","L":"lower"}; out=[]
    if len(p)>=1: out.append(m1.get(p[0].upper(),p[0]))
    if len(p)>=2: out.append(m2.get(p[1].upper(),p[1]))
    if len(p)>=3: out.append(m3.get(p[2].upper(),p[2]))
    return "_".join(out) if out else None


def parse_mendeley(raw):
    raw = Path(raw); rows = []
    for wav in sorted(raw.rglob("*.wav")):
        stem = wav.stem
        if "_" not in stem: continue
        pre, rest = stem.split("_", 1); f = [x.strip() for x in rest.split(",")]
        ev, ph = _mp(f[1] if len(f) > 1 else "N")
        num = re.sub(r"\D","",pre) or pre
        rows.append({"audio_path":str(wav),"dataset":"mendeley","patient_id":f"mendeley_{num}",
                     "start":0.0,"end":0.0,"event":ev,"phase":ph,
                     "diagnosis":_md(f[0]) if f else None,"severity":None,
                     "location":_ml(f[2]) if len(f)>2 else None,
                     "filter_type":_MF.get(pre[0].upper(), pre[0]) if pre else None})
    return rows


def parse_hf_lung(raw):
    raw = Path(raw); rows = []; CONT = {"wheeze","rhonchi","stridor"}
    for txt in sorted(raw.rglob("*_label.txt")):
        types = set()
        for line in txt.read_text().splitlines():
            p = line.split()
            if p: types.add(p[0].strip().lower())
        hc = "d" in types; hw = bool(types & CONT)
        ev = "both" if hw and hc else ("wheeze" if hw else ("crackle" if hc else "normal"))
        stem = txt.name[:-len("_label.txt")] if txt.name.endswith("_label.txt") else txt.stem
        rows.append({"audio_path":str(txt.with_name(stem+".wav")),"dataset":"hf_lung",
                     "patient_id":f"hflung_{stem}","start":0.0,"end":0.0,"event":ev,"phase":None,
                     "diagnosis":None,"severity":None,"location":None,"filter_type":None})
    return rows


def parse_sprsound(raw):
    raw = Path(raw); rows = []
    for jf in sorted(raw.rglob("*.json")):
        try: data = json.loads(jf.read_text())
        except Exception: continue
        ev_ann = data.get("event_annotation") or data.get("events") or []
        types = [str(e.get("type","")).lower() for e in ev_ann]; rec = str(data.get("record_annotation","")).lower()
        hw = any(("wheez" in t or "rhonchi" in t or "stridor" in t) for t in types) or "cas" in rec
        hc = any("crackle" in t for t in types) or "das" in rec
        ev = "both" if hw and hc else ("wheeze" if hw else ("crackle" if hc else "normal"))
        pid = jf.stem.split("_")[0] if "_" in jf.stem else jf.stem
        rows.append({"audio_path":str(jf.with_suffix(".wav")),"dataset":"sprsound",
                     "patient_id":f"sprsound_{pid}","start":0.0,"end":0.0,"event":ev,"phase":None,
                     "diagnosis":None,"severity":None,"location":None,"filter_type":None})
    return rows


PARSERS = {"icbhi":parse_icbhi,"mendeley":parse_mendeley,"hf_lung":parse_hf_lung,"sprsound":parse_sprsound}


def build_manifest(specs):
    rows = []
    for name, path in specs:
        r = PARSERS[name](Path(path)); print(f"[ingest] {name}: {len(r)} rows from {path}"); rows.extend(r)
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


def write_manifest(df, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        try: df.to_parquet(out, index=False); return out
        except Exception as e:
            out = out.with_suffix(".csv"); print(f"[ingest] parquet unavailable ({e}); CSV -> {out}")
    df.to_csv(out, index=False); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="data/raw")
    ap.add_argument("--datasets", nargs="+", default=["icbhi"],
                    help="names, optionally name=PATH e.g. hf_lung=data/raw/HF_Lung")
    ap.add_argument("--out", default="data/processed/manifests/all.parquet")
    a = ap.parse_args()
    specs = []
    for d in a.datasets:
        if "=" in d: name, path = d.split("=", 1)
        else: name, path = d, str(Path(a.raw_root) / d)
        specs.append((name, path))
    df = build_manifest(specs)
    print("[ingest] wrote", len(df), "rows ->", write_manifest(df, Path(a.out)))
    if len(df): print(df.groupby("dataset")["event"].value_counts())


if __name__ == "__main__":
    main()
