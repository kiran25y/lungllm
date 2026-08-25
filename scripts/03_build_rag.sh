#!/usr/bin/env bash
set -e
python -m lungllm.rag.build_index --reports data/rag/reports/reports.jsonl
python -m lungllm.training.copo --reports data/rag/reports/reports.jsonl
