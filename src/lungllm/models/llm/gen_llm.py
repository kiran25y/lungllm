# REPO PATH: src/lungllm/models/llm/gen_llm.py
"""Open-LLM multimodal generative head (Qwen2.5, ungated). Prepends audio prefix tokens
to text embeddings via inputs_embeds so the LLM generates a report conditioned on audio."""
from __future__ import annotations
import torch
import torch.nn as nn

DEFAULT_LLM = "Qwen/Qwen2.5-3B-Instruct"


class GenLLM(nn.Module):
    def __init__(self, model_name=DEFAULT_LLM, lora_r=16, lora_alpha=32, dtype="bfloat16"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))
        try:
            from peft import LoraConfig, get_peft_model
            self.llm = get_peft_model(self.llm, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        except Exception as e:
            print("peft unavailable:", e)
        self.hidden = self.llm.config.hidden_size

    def _embed(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, audio_prefix, input_ids, attention_mask, labels=None):
        B, k, _ = audio_prefix.shape
        tok_emb = self._embed(input_ids)
        inputs_embeds = torch.cat([audio_prefix.to(tok_emb.dtype), tok_emb], dim=1)
        pmask = torch.ones(B, k, device=attention_mask.device, dtype=attention_mask.dtype)
        attn = torch.cat([pmask, attention_mask], dim=1)
        out_labels = None
        if labels is not None:
            plab = torch.full((B, k), -100, device=labels.device, dtype=labels.dtype)
            out_labels = torch.cat([plab, labels], dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=out_labels)

    @torch.no_grad()
    def generate(self, audio_prefix, prompt_ids, prompt_mask, max_new_tokens=48):
        B, k, _ = audio_prefix.shape
        tok_emb = self._embed(prompt_ids)
        inputs_embeds = torch.cat([audio_prefix.to(tok_emb.dtype), tok_emb], dim=1)
        pmask = torch.ones(B, k, device=prompt_mask.device, dtype=prompt_mask.dtype)
        attn = torch.cat([pmask, prompt_mask], dim=1)
        gen = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                                max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.batch_decode(gen, skip_special_tokens=True)