#!/usr/bin/env bash
set -e
# The working AST baseline (reaches ~0.65 ICBHI). Fine-tune, class-balanced.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m lungllm.training.train_multitask \
  --train data/splits/train.parquet --val data/splits/val.parquet \
  --epochs 12 --batch_size 8 --lr 2e-5 --finetune --balanced_sampler --num_workers 0 \
  --out checkpoints/sft/ast_multitask.pt
