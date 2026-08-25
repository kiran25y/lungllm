# REPO PATH: src/lungllm/models/multimodal_model.py
"""Phase 2/3 — full v2 assembly: aligned encoder -> bridge -> (MoE) -> MedGemma + heads.

Forward path (now wired):
  waveforms -> AST encoder -> AudioMapper bridge -> Sparse MoE (EAM logged)
            -> MedGemmaMultimodal( audio_prefix, [+retrieved text], instruction+target )
            -> {anomaly, disease, lm_loss, moe_info}
Retrieval (optional) injects top-k retrieved report strings as extra prefix embeddings.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .pretrained_encoder import ASTEncoder
from .bridge.audio_mapper import AudioMapper
from .moe.sparse_moe import SparseMoE
from .llm.medgemma_wrapper import MedGemmaMultimodal


class LungLLMMoEv2(nn.Module):
    def __init__(self, aligned_encoder_ckpt=None, use_moe=True, k_prefix=4,
                 num_anomaly=4, num_disease=0, retriever=None, top_k_reports=3,
                 freeze_encoder=False, llm_name="google/medgemma-4b-it"):
        super().__init__()
        self.encoder = ASTEncoder(freeze=freeze_encoder)
        if aligned_encoder_ckpt:
            sd = torch.load(aligned_encoder_ckpt, map_location="cpu")
            self.encoder.load_state_dict(sd.get("encoder", sd), strict=False)
        self.llm = MedGemmaMultimodal(model_name=llm_name, num_anomaly=num_anomaly,
                                      num_disease=num_disease)
        self.bridge = AudioMapper(self.encoder.embed_dim, self.llm.hidden, k=k_prefix)
        self.moe = SparseMoE(dim=self.llm.hidden, num_experts=5, top_k=2) if use_moe else None
        self.retriever = retriever
        self.top_k_reports = top_k_reports

    def _prefix(self, waveforms):
        z = self.encoder(waveforms)["clip_embedding"]      # [B, D]
        prefix = self.bridge(z)                            # [B, k, hidden]
        aux, moe_info = None, None
        if self.moe is not None:
            prefix, aux, moe_info = self.moe(prefix)
        return prefix, aux, moe_info

    def _retrieved_embeds(self, symptom_text, device):
        if self.retriever is None or symptom_text is None:
            return None, None
        joined = []
        for t in symptom_text:
            hits = self.retriever.retrieve(t)
            texts = [(h[0] if isinstance(h, (list, tuple)) else str(h))
                     for h in hits[:self.top_k_reports]]
            joined.append(" ".join(texts) if texts else "")
        return self.llm.embed_text(joined, device)

    def forward(self, waveforms, input_ids=None, attention_mask=None, labels=None,
                symptom_text=None):
        prefix, aux, moe_info = self._prefix(waveforms)
        extra_embeds, extra_mask = self._retrieved_embeds(symptom_text, prefix.device)
        out = self.llm(prefix, input_ids=input_ids, attention_mask=attention_mask,
                       labels=labels, extra_embeds=extra_embeds, extra_mask=extra_mask)
        if isinstance(out, dict):
            out["moe_info"] = moe_info
            if aux is not None:
                out["moe_aux"] = aux
        return out

    @torch.no_grad()
    def generate(self, waveforms, prompt_ids, prompt_mask, max_new_tokens=64,
                 symptom_text=None):
        prefix, _, _ = self._prefix(waveforms)
        extra_embeds, extra_mask = self._retrieved_embeds(symptom_text, prefix.device)
        return self.llm.generate(prefix, prompt_ids, prompt_mask,
                                 max_new_tokens=max_new_tokens,
                                 extra_embeds=extra_embeds, extra_mask=extra_mask)