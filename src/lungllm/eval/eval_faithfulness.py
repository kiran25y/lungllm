# REPO PATH: src/lungllm/eval/eval_faithfulness.py
"""EXP6 — the faithfulness result: generate reports from each checkpoint on held-out clips
and score acoustic faithfulness (fraction of acoustic claims supported by the true event).
Compares SFT vs CoPO (add BERTScore-DPO to complete the table)."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import pandas as pd, torch
from lungllm.data.features import load_audio
from lungllm.models.gen_model import GenModel

KW = {"wheeze": ["wheez"], "crackle": ["crackl", "crepit", "rale"],
      "normal": ["no adventitious", "normal", "clear"]}


def detected(ev):
    ev = str(ev)
    if ev == "both": return {"wheeze", "crackle"}
    if ev in ("wheeze", "crackle"): return {ev}
    return {"normal"}


def factscore(report, ev):
    t = report.lower(); det = detected(ev); claims = 0; sup = 0
    for lab, kws in KW.items():
        if any(k in t for k in kws):
            claims += 1
            if lab in det: sup += 1
    return sup / claims if claims else 1.0


def load_clips(manifest, n, sr=16000, maxs=10.0):
    p = Path(manifest)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.sample(min(n, len(df)), random_state=0).to_dict("records")
    out = []
    for r in df:
        w = load_audio(r["audio_path"], sr)
        s, e = int(float(r["start"]) * sr), int(float(r["end"]) * sr)
        if e > s: w = w[s:e]
        if w.numel() > int(maxs*sr): w = w[:int(maxs*sr)]
        if w.numel() < int(0.5*sr): w = torch.nn.functional.pad(w, (0, int(0.5*sr)-w.numel()))
        out.append((w, r.get("event")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/splits/icbhi_official_test.parquet")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="label=path pairs, e.g. sft=checkpoints/gen/sft_gen.pt copo=checkpoints/gen/copo.pt")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--encoder_ckpt", default="checkpoints/aligned/ast_aligned_clean.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    clips = load_clips(a.manifest, a.n)
    print(f"[faithfulness] {len(clips)} held-out clips from {a.manifest}")
    for spec in a.ckpts:
        label, path = spec.split("=", 1)
        model = GenModel(a.llm, encoder_ckpt=a.encoder_ckpt if Path(a.encoder_ckpt).exists() else None)
        model.load_state_dict(torch.load(path, map_location="cpu")["model"], strict=False)
        model.to(a.device).eval()
        tok = model.llm.tok
        pr = tok(["Auscultation"], return_tensors="pt")
        scores = []
        with torch.no_grad():
            for w, ev in clips:
                gen = model.generate([w], pr["input_ids"].to(a.device),
                                     pr["attention_mask"].to(a.device), max_new_tokens=24)
                rep = gen[0] if gen else ""
                scores.append(factscore(rep, ev))
        print(f"  {label:12s} Acoustic FActScore = {sum(scores)/max(len(scores),1):.3f}")


if __name__ == "__main__":
    main()