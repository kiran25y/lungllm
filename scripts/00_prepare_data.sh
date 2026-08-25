#!/usr/bin/env bash
set -e
# Build the unified manifest from all 4 datasets (edit paths to match configs/paths.yaml)
python -m lungllm.data.ingest \
  --datasets icbhi=data/raw/icbhi mendeley=data/raw/mendeley \
             hf_lung=data/raw/HF_Lung sprsound=data/raw/new_sprs/train \
  --out data/processed/manifests/all.parquet
# Patient-aware split
python -m lungllm.data.splits --manifest data/processed/manifests/all.parquet --out_dir data/splits
# (optional) OOD + unseen-disease splits
python -m lungllm.data.splits_v2 --manifest data/processed/manifests/all.parquet --mode ood
python -m lungllm.data.splits_v2 --manifest data/processed/manifests/all.parquet --mode unseen_disease
