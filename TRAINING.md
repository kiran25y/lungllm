# How to train — LungLLM-MoE

## 0. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # reads requirements.txt
```

## 1. Point the config at your data
Edit **`configs/paths.yaml`** so the four `raw:` paths match where your datasets live:
```
raw:
  icbhi:    data/raw/icbhi           # *.wav + *.txt + patient_diagnosis.csv
  hf_lung:  data/raw/HF_Lung         # *.wav + *_label.txt
  sprsound: data/raw/new_sprs/train  # *.wav + *.json
  mendeley: data/raw/mendeley        # Audio Files/*.wav
```
`patient_diagnosis.csv` and `ICBHI_challenge_train_test.txt` must be inside the ICBHI folder
for disease labels + the official split (download from the ICBHI / Kaggle release if missing).

## 2. Build manifest + splits  (Phase 0)
```bash
bash scripts/00_prepare_data.sh
# -> data/processed/manifests/all.parquet
# -> data/splits/{train,val,test}.parquet  (+ ood_/unseen_ splits)
```
Add ICBHI diagnoses (if patient_diagnosis.csv is present it's picked up automatically).
Official citable split:
```bash
python -m lungllm.data.official_split \
  --manifest data/processed/manifests/all.parquet \
  --split_file data/raw/icbhi/ICBHI_challenge_train_test.txt --out_dir data/splits
```

## 3. Train the working baseline  (AST, reaches ~0.65 ICBHI)
```bash
bash scripts/01_train_baseline.sh
```
Watch `ICBHI` and the `NWCB` per-class recalls per epoch. This is the model you already have.

## 4. Phase 1 — encoder alignment (biggest upgrade)
```bash
bash scripts/02_align_encoder.sh
# builds data/rag/reports/reports.jsonl, then trains the CKA+InfoNCE-aligned encoder
# -> checkpoints/aligned/ast_aligned.pt
```
Then re-train the baseline initialised from the aligned encoder (expected AUROC/zero-shot gain).

## 5. Phase 4/5 — RAG + faithfulness data
```bash
bash scripts/03_build_rag.sh
# -> data/rag/index/reports.faiss  + acoustic-grounded preference pairs
```

## 6. Evaluate
```bash
python -c "from lungllm.eval.evaluate_v2 import icbhi_score, macro_f1; print('eval utils ready')"
```
Use `lungllm.eval.evaluate_v2` (ICBHI Score / macro-F1 / balanced-acc), `lungllm.eval.faithfulness`
(Acoustic FActScore), and `lungllm.eval.moe_activation_map` (Expert-Activation Map).

## Phases still requiring wiring (need MedGemma weights on GPU)
- `models/llm/medgemma_wrapper.py` — one `inputs_embeds` concat TODO
- `models/multimodal_model.py`, `training/sft_multitask_v2.py`, `agent/orchestrator.py`
These are the generative/agent stages; the classifier + alignment + RAG + eval all run without them.

## Which files are real vs. scaffold
Real & runnable: data/*, models/pretrained_encoder, models/moe, models/alignment/*,
training/{engine,train_pretrained,train_multitask,align_encoder,copo}, rag/*, eval/*, data/lungmix.
Scaffold (finish the TODO): models/llm/medgemma_wrapper, models/multimodal_model,
training/sft_multitask_v2, agent/orchestrator.
