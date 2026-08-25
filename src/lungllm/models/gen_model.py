# REPO PATH: src/lungllm/models/gen_model.py
"""Generative model: (aligned) AST encoder -> AudioMapper bridge -> open LLM."""
from __future__ import annotations
import torch
import torch.nn as nn
from .pretrained_encoder import ASTEncoder
from .bridge.audio_mapper import AudioMapper
from .llm.gen_llm import GenLLM, DEFAULT_LLM


class GenModel(nn.Module):
    def __init__(self, llm_name=DEFAULT_LLM, encoder_ckpt=None, k_prefix=4, freeze_encoder=True):
        super().__init__()
        self.encoder = ASTEncoder(freeze=freeze_encoder)
        if encoder_ckpt:
            ck = torch.load(encoder_ckpt, map_location="cpu")
            self.encoder.load_state_dict(ck.get("encoder", ck), strict=False)
        self.llm = GenLLM(llm_name)
        self.bridge = AudioMapper(self.encoder.embed_dim, self.llm.hidden, k=k_prefix)

    def audio_prefix(self, waveforms):
        z = self.encoder(waveforms)["clip_embedding"]
        return self.bridge(z)

    def forward(self, waveforms, input_ids, attention_mask, labels=None):
        return self.llm(self.audio_prefix(waveforms), input_ids, attention_mask, labels)

    @torch.no_grad()
    def generate(self, waveforms, prompt_ids, prompt_mask, max_new_tokens=48):
        return self.llm.generate(self.audio_prefix(waveforms), prompt_ids, prompt_mask, max_new_tokens)