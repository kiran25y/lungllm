# REPO PATH: src/lungllm/training/train_grpo.py
"""EXP5 — GRPO: reinforcement learning with a VERIFIABLE faithfulness reward (RLVR).

For each clip, sample G candidate reports, score each with the Acoustic FActScore reward
(fraction of acoustic claims supported by the true event), and update the policy with a
group-relative advantage. On-policy (fixes CoPO's off-policy drift) and directly maximizes
the faithfulness metric. Start from the SFT checkpoint.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from lungllm.data.features import load_audio
from lungllm.models.gen_model import GenModel
# add imports
from lungllm.eval.sed_factscore import acoustic_factscore
from lungllm.eval.sed_grounding import clip_detections_from_label
from lungllm.data.sed_records import _label_path_for

# dataset __getitem__: return the row's audio_path + dataset instead of just event
return w, r["audio_path"], str(r.get("dataset")), str(r.get("event"))

# reward: prefer typed label grounding, fall back to clip event
def clip_reward_dets(audio_path, dataset, event):
    lp = _label_path_for(audio_path, dataset)
    if lp is not None and lp.exists():
        return clip_detections_from_label(str(lp), dataset, with_phase=True)   # typed+phase truth
    return None, event   # fallback handled by acoustic_factscore(event=...)
# in the loop: reward = acoustic_factscore(text, detections=dets, check_phase=True)

KW = {"wheeze": ["wheez"], "crackle": ["crackl", "crepit", "rale"], "normal": ["no adventitious", "normal", "clear"]}
def detected(ev):
    ev = str(ev)
    if ev == "both": return {"wheeze", "crackle"}
    if ev in ("wheeze", "crackle"): return {ev}
    return {"normal"}
def factscore(rep, ev):
    t = rep.lower(); det = detected(ev); c = 0; s = 0
    for lab, kws in KW.items():
        if any(k in t for k in kws):
            c += 1; s += (lab in det)
    return s / c if c else 0.5      # no claim -> neutral (discourages empty reports)


class Clips(Dataset):
    def __init__(self, manifest, sr=16000, maxs=10.0):
        p = Path(manifest); df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        self.rows = df.to_dict("records"); self.sr = sr; self.mx = int(maxs*sr); self.mn = int(0.5*sr)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; w = load_audio(r["audio_path"], self.sr)
        s, e = int(float(r["start"])*self.sr), int(float(r["end"])*self.sr)
        if e > s: w = w[s:e]
        if w.numel() > self.mx: w = w[:self.mx]
        if w.numel() < self.mn: w = torch.nn.functional.pad(w, (0, self.mn-w.numel()))
        return w, str(r.get("event"))


def collate(b): return [x[0] for x in b], [x[1] for x in b]


def seq_logp_ids(model, prefix, ids, device):
    """log-prob of token ids given a fixed audio prefix [1,k,H]; ids [G,T]."""
    G, T = ids.shape
    pref = prefix.expand(G, -1, -1)
    out = model.llm(pref, ids.to(device), torch.ones_like(ids).to(device), labels=None)
    k = pref.size(1); lsm = torch.log_softmax(out.logits.float(), dim=-1)
    pred = lsm[:, k-1:k-1+T, :]
    return pred.gather(-1, ids.to(device).unsqueeze(-1)).squeeze(-1).sum(1)   # [G]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/splits/train_official_clean.parquet")
    ap.add_argument("--sft_ckpt", default="checkpoints/gen/sft_gen.pt")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--encoder_ckpt", default="checkpoints/aligned/ast_aligned_clean.pt")
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=24); ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=2000, help="clips per epoch (RL is slow)")
    ap.add_argument("--out", default="checkpoints/gen/grpo.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("no CUDA")
    model = GenModel(a.llm, encoder_ckpt=a.encoder_ckpt if Path(a.encoder_ckpt).exists() else None)
    if Path(a.sft_ckpt).exists():
        model.load_state_dict(torch.load(a.sft_ckpt, map_location="cpu")["model"], strict=False)
    model.to(a.device); tok = model.llm.tok
    prompt = tok(["Auscultation"], return_tensors="pt")
    dl = DataLoader(Clips(a.manifest), batch_size=1, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[grpo] group {a.group} | reward=Acoustic FActScore | limit {a.limit}", flush=True)
    for ep in range(1, a.epochs+1):
        model.train(); run_r = 0.0; n = 0; t0 = time.time()
        for bi, (waves, evs) in enumerate(dl, 1):
            if bi > a.limit: break
            ev = evs[0]
            pref = model.audio_prefix(waves).detach()              # [1,k,H]
            with torch.no_grad():
                pe = pref.expand(a.group, -1, -1)
                pids = prompt["input_ids"].expand(a.group, -1).to(a.device)
                gen = model.llm.llm.generate(inputs_embeds=torch.cat(
                    [pe.to(model.llm._embed(pids).dtype), model.llm._embed(pids)], dim=1),
                    attention_mask=torch.ones(a.group, a.group and pe.size(1)+pids.size(1)).to(a.device),
                    do_sample=True, temperature=a.temp, max_new_tokens=a.max_new)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            rewards = torch.tensor([factscore(t, ev) for t in texts], device=a.device)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
            logp = seq_logp_ids(model, pref, gen, a.device)
            loss = -(adv * logp).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run_r += rewards.mean().item(); n += 1
            if bi % a.log_every == 0:
                print(f"  ep{ep} {bi}/{min(a.limit,len(dl))} mean_reward {run_r/max(n,1):.3f} {(time.time()-t0)/bi:.2f}s/clip", flush=True)
        print(f"epoch {ep:02d} mean_reward {run_r/max(n,1):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, a.out)
    print("saved GRPO model ->", a.out, flush=True)


if __name__ == "__main__":
    main()