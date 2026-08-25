# REPO PATH: src/lungllm/models/llm/medgemma_wrapper.py
"""Phase 2 — LoRA MedGemma-4B multimodal head.

Prepends audio prefix tokens (+ optional retrieved-report / symptom text embeddings) to
the token stream, then runs the LLM. SFT trains LoRA + bridge; classification heads read
the pooled hidden state.

Fused sequence (left -> right):
    [ audio_prefix (k) ] [ retrieved/symptom text (r) ] [ instruction+target text (t) ]
Only target-text positions carry LM labels; everything left of them is masked (-100).
Classification heads read a masked-mean pool over the non-audio positions (falls back to
the audio prefix when no text is supplied, e.g. pure classification).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MedGemmaMultimodal(nn.Module):
    def __init__(self, model_name="google/medgemma-4b-it", lora_r=16, lora_alpha=32,
                 load_in_4bit=True, num_anomaly=4, num_disease=0):
        super().__init__()
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, token=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = self._load_backbone(model_name, load_in_4bit)
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if getattr(self, 'used_4bit', False):
                self.llm = prepare_model_for_kbit_training(self.llm)
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM")
            self.llm = get_peft_model(self.llm, cfg)
        except Exception as e:
            print("peft unavailable:", e)
        self.hidden = self._infer_hidden()
        self.anomaly_head = nn.Linear(self.hidden, num_anomaly)
        self.disease_head = nn.Linear(self.hidden, num_disease) if num_disease > 0 else None

    def _load_backbone(self, model_name, load_in_4bit):
        from transformers import AutoModelForCausalLM
        import importlib.util
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        self.used_4bit = bool(load_in_4bit and has_bnb)
        kw = dict(device_map="auto", torch_dtype=torch.bfloat16, token=True)
        if load_in_4bit and not has_bnb:
            print("bitsandbytes not installed -> loading in bf16 (no 4-bit).")
        if self.used_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **kw)
        except Exception as e:
            print("CausalLM load failed, trying ImageTextToText backbone:", e)
            from transformers import AutoModelForImageTextToText
            full = AutoModelForImageTextToText.from_pretrained(model_name, **kw)
            return getattr(full, "language_model", full)

    def _infer_hidden(self):
        cfg = self.llm.config
        for attr in ("hidden_size", "text_config"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int):
                return v
            if v is not None and hasattr(v, "hidden_size"):
                return v.hidden_size
        return self.llm.get_input_embeddings().embedding_dim

    def _embed(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def embed_text(self, strings, device):
        t = self.tok(strings, return_tensors="pt", padding=True, truncation=True,
                     max_length=128).to(device)
        return self._embed(t["input_ids"]), t["attention_mask"]

    def forward(self, audio_prefix, input_ids=None, attention_mask=None, labels=None,
                extra_embeds=None, extra_mask=None):
        """audio_prefix: [B, k, hidden]. input_ids/attention_mask/labels: instruction+target
        text (optional). extra_embeds/extra_mask: retrieved/symptom embeddings (optional).
        Returns dict {anomaly, disease?, lm_loss?, logits?, pooled}."""
        emb_dtype = self.llm.get_input_embeddings().weight.dtype
        device = audio_prefix.device
        B, k, _ = audio_prefix.shape
        parts = [audio_prefix.to(emb_dtype)]
        masks = [torch.ones(B, k, device=device, dtype=torch.long)]
        label_parts = [torch.full((B, k), -100, device=device, dtype=torch.long)]
        text_start = k

        if extra_embeds is not None:
            parts.append(extra_embeds.to(emb_dtype))
            em = extra_mask if extra_mask is not None else torch.ones(
                B, extra_embeds.size(1), device=device, dtype=torch.long)
            masks.append(em)
            label_parts.append(torch.full(extra_embeds.shape[:2], -100, device=device, dtype=torch.long))

        if input_ids is not None:
            parts.append(self._embed(input_ids).to(emb_dtype))
            tm = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            masks.append(tm)
            label_parts.append(labels if labels is not None else torch.full_like(input_ids, -100))

        inputs_embeds = torch.cat(parts, dim=1)
        attn = torch.cat(masks, dim=1)
        out_labels = torch.cat(label_parts, dim=1) if labels is not None else None

        res = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn,
                       labels=out_labels, output_hidden_states=True)
        last = res.hidden_states[-1]

        pool_mask = attn.clone()
        pool_mask[:, :text_start] = 0
        if pool_mask.sum() == 0:
            pool_mask = attn.clone()
        pm = pool_mask.unsqueeze(-1).to(last.dtype)
        pooled = (last * pm).sum(1) / pm.sum(1).clamp(min=1.0)
        pooled = pooled.to(self.anomaly_head.weight.dtype)

        out = {"anomaly": self.anomaly_head(pooled), "pooled": pooled}
        if self.disease_head is not None:
            out["disease"] = self.disease_head(pooled)
        if out_labels is not None:
            out["lm_loss"] = res.loss
            out["logits"] = res.logits
        return out

    @torch.no_grad()
    def generate(self, audio_prefix, prompt_ids, prompt_mask, max_new_tokens=64,
                 extra_embeds=None, extra_mask=None):
        emb_dtype = self.llm.get_input_embeddings().weight.dtype
        device = audio_prefix.device
        B, k, _ = audio_prefix.shape
        parts = [audio_prefix.to(emb_dtype)]
        masks = [torch.ones(B, k, device=device, dtype=torch.long)]
        if extra_embeds is not None:
            parts.append(extra_embeds.to(emb_dtype))
            masks.append(extra_mask if extra_mask is not None else
                         torch.ones(B, extra_embeds.size(1), device=device, dtype=torch.long))
        parts.append(self._embed(prompt_ids).to(emb_dtype))
        masks.append(prompt_mask)
        gen = self.llm.generate(inputs_embeds=torch.cat(parts, dim=1),
                                attention_mask=torch.cat(masks, dim=1),
                                max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.batch_decode(gen, skip_special_tokens=True)