# REPO PATH: src/lungllm/training/sft_multitask_v2.py
"""Phase 2 — multi-task instruction SFT for the v2 model (scaffold).

Trains bridge + MoE + LoRA + heads on the instruction corpus (classify/detect/report/
reason/DDx/compare/locate). Reuses your DS/collate patterns. The LM loss + head losses
combine; MoE aux + class weights as before. Fill the train loop once
MedGemmaMultimodal.forward is wired.
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train_all_dx.parquet")
    ap.add_argument("--val", default="data/splits/icbhi_official_test.parquet")
    ap.add_argument("--instructions", default="data/processed/instructions.jsonl")
    ap.add_argument("--aligned_encoder", default="checkpoints/aligned/ast_aligned.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/sft/v2.pt")
    a = ap.parse_args()
    print("[sft_v2] scaffold — instantiate LungLLMMoEv2(aligned_encoder_ckpt=...),")
    print("        build instruction batches, combine LM loss + anomaly/disease/MoE losses,")
    print("        train LoRA + bridge + MoE + heads. See multimodal_model.py.")


if __name__ == "__main__":
    main()
