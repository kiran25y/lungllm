#!/usr/bin/env bash
set -e
# Phase 1: report corpus, then CKA+InfoNCE encoder alignment (highest-leverage upgrade)
python -m lungllm.rag.build_reports --manifest data/processed/manifests/all.parquet --mode template
python -m lungllm.training.align_encoder \
  --reports data/rag/reports/reports.jsonl --out checkpoints/aligned/ast_aligned.pt
