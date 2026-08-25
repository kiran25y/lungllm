from __future__ import annotations
import argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from lungllm.training.train_multitask import DS, collate, evaluate as mt_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+",
                    default=["data/splits/val.parquet", "data/splits/test.parquet",
                             "data/splits/ood_test.parquet"])
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); dv = ck.get("disease_vocab", {})
    from lungllm.models.pretrained_encoder import ASTEncoder, PretrainedMultiTaskModel
    if a.moe:
        from lungllm.models.moe_model import PretrainedMoEModel
        model = PretrainedMoEModel(ASTEncoder(freeze=False), num_anomaly=4, num_disease=len(dv))
    else:
        model = PretrainedMultiTaskModel(ASTEncoder(freeze=False), num_anomaly=4, num_disease=len(dv))
    model.load_state_dict(ck["model"]); model.to(a.device).eval()
    print(f"{'split':44s} {'ICBHI':>7s} {'Se':>6s} {'Sp':>6s} {'dis_acc':>8s}")
    for sp in a.splits:
        m = mt_eval(model, DataLoader(DS(sp, dv), batch_size=8, shuffle=False, collate_fn=collate))
        da = m['da'] if m['da'] is not None else float('nan')
        print(f"{sp:44s} {m['icbhi']:7.4f} {m['se']:6.3f} {m['sp']:6.3f} {da:8.4f}")


if __name__ == "__main__":
    main()
