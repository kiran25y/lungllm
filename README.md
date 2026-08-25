# LungLLM-MoE (v2)

Interpretable, multimodal respiratory-sound analysis. A pretrained **AST** encoder
(fine-tuned on ICBHI + HF Lung + SPRSound + Mendeley → ~0.65 ICBHI Score, multi-task
disease head) upgraded toward an aligned, interpretable, agentic system.

```
configs/        paths.yaml (dataset folders) · data.yaml · train.yaml
data/           raw/{icbhi,hf_lung,sprsound,mendeley} · processed · splits · rag
scripts/        00_prepare_data · 01_train_baseline · 02_align_encoder · 03_build_rag
src/lungllm/
  data/         ingest (4 parsers) · features · dataset · augment · splits · lungmix · build_text
  models/       pretrained_encoder (AST) · moe · alignment · bridge · llm · multimodal_model
  training/     engine · train_pretrained · train_multitask · align_encoder · copo · sft_multitask_v2
  rag/          build_reports · build_index · retriever
  eval/         evaluate_v2 · faithfulness · moe_activation_map
  agent/        orchestrator
```

**Start here:** `TRAINING.md`. Install with `pip install -e .`, edit `configs/paths.yaml`,
then run `scripts/00…03`.
