# REPO PATH: src/lungllm/models/multimodal_model.py
"""Phase 2/3 — full v2 assembly: aligned encoder -> bridge -> (MoE) -> MedGemma + heads.

Scaffold: constructs and holds all submodules with correct dims. The forward wiring that
concatenates audio prefix + text + retrieved reports into the LLM lives in
MedGemmaMultimodal.forward (marked TODO there). MoE + Expert-Activation Map are logged.
"""
from __future__ import annotations
import torch.nn as nn

from .pretrained_encoder import ASTEncoder
from .bridge.audio_mapper import AudioMapper
from .moe.sparse_moe import SparseMoE
from .llm.medgemma_wrapper import MedGemmaMultimodal


class LungLLMMoEv2(nn.Module):
    def __init__(self, aligned_encoder_ckpt=None, use_moe=True, k_prefix=4,
                 num_anomaly=4, num_disease=0, retriever=None):
        super().__init__()
        self.encoder = ASTEncoder(freeze=False)
        if aligned_encoder_ckpt:
            import torch
            sd = torch.load(aligned_encoder_ckpt, map_location="cpu")
            self.encoder.load_state_dict(sd["encoder"], strict=False)
        self.llm = MedGemmaMultimodal(num_anomaly=num_anomaly, num_disease=num_disease)
        self.bridge = AudioMapper(self.encoder.embed_dim, self.llm.hidden, k=k_prefix)
        self.moe = SparseMoE(dim=self.llm.hidden, num_experts=5, top_k=2) if use_moe else None
        self.retriever = retriever

    def forward(self, waveforms, text=None, labels=None):
        z = self.encoder(waveforms)["clip_embedding"]          # [B, D]
        prefix = self.bridge(z)                                # [B, k, hidden]
        moe_info = None
        if self.moe is not None:
            prefix, aux, moe_info = self.moe(prefix)
        # retrieval (optional)
        rag_text = None
        if self.retriever is not None and text is not None:
            rag_text = [self.retriever.retrieve(t) for t in text]
        out = self.llm(prefix, text=text, labels=labels)       # TODO wiring in wrapper
        if isinstance(out, dict):
            out["moe_info"] = moe_info; out["rag"] = rag_text
        return out
