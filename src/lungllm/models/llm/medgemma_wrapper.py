# REPO PATH: src/lungllm/models/llm/medgemma_wrapper.py
"""Phase 2 — LoRA MedGemma-4B multimodal head.

Prepends audio prefix tokens (+ optional retrieved-report / symptom text embeddings) to
the token stream, then runs the LLM. SFT trains LoRA + bridge; classification heads read
the pooled hidden state. This is a working scaffold — fill in the forward wiring for your
transformers version (inputs_embeds concatenation is the key step, marked TODO).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MedGemmaMultimodal(nn.Module):
    def __init__(self, model_name="google/medgemma-4b-it", lora_r=16, lora_alpha=32,
                 load_in_4bit=True, num_anomaly=4, num_disease=0):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_4bit=load_in_4bit, device_map="auto")
        try:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            self.llm = get_peft_model(self.llm, cfg)
        except Exception as e:
            print("peft unavailable:", e)
        self.hidden = self.llm.config.hidden_size
        self.anomaly_head = nn.Linear(self.hidden, num_anomaly)
        self.disease_head = nn.Linear(self.hidden, num_disease) if num_disease > 0 else None

    def embed_text(self, strings, device):
        t = self.tok(strings, return_tensors="pt", padding=True, truncation=True,
                     max_length=128).to(device)
        emb = self.llm.get_input_embeddings()(t["input_ids"])
        return emb, t["attention_mask"]

    def forward(self, audio_prefix, text=None, labels=None):
        """audio_prefix: [B, k, hidden] from the bridge.
        TODO: concat [audio_prefix ; text_emb ; instruction_emb] as inputs_embeds,
        build attention_mask, run self.llm(inputs_embeds=..., labels=... for gen),
        pool last hidden state for the classification heads. Returns dict of logits + LM loss.
        """
        raise NotImplementedError("Wire inputs_embeds concatenation for your transformers version.")
