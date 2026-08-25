# REPO PATH: src/lungllm/models/pretrained_encoder.py
"""Pretrained AST encoder + single-task and multi-task heads."""
from __future__ import annotations
import torch
import torch.nn as nn

DEFAULT_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTEncoder(nn.Module):
    def __init__(self, model_name=DEFAULT_AST, freeze=True, sample_rate=16000):
        super().__init__()
        from transformers import ASTModel, ASTFeatureExtractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.ast = ASTModel.from_pretrained(model_name)
        self.embed_dim = self.ast.config.hidden_size
        self.sample_rate = sample_rate; self.frozen = freeze
        if freeze:
            self.ast.eval()
            for p in self.ast.parameters(): p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen: self.ast.eval()
        return self

    def forward(self, waveforms):
        if torch.is_tensor(waveforms):
            wl = [w.detach().float().cpu().numpy() for w in waveforms]
        else:
            wl = [(w.detach().float().cpu().numpy() if torch.is_tensor(w) else w) for w in waveforms]
        feats = self.feature_extractor(wl, sampling_rate=self.sample_rate, return_tensors="pt")
        iv = feats["input_values"].to(next(self.ast.parameters()).device)
        if self.frozen:
            with torch.no_grad(): out = self.ast(iv)
        else:
            out = self.ast(iv)
        return {"clip_embedding": out.pooler_output}


class PretrainedAnomalyModel(nn.Module):
    def __init__(self, encoder, num_anomaly=4, hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.embed_dim, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, num_anomaly))

    def forward(self, waveforms):
        z = self.encoder(waveforms)["clip_embedding"]
        return {"logits": self.head(z), "aux_loss": z.new_zeros(())}


class PretrainedMultiTaskModel(nn.Module):
    def __init__(self, encoder, num_anomaly=4, num_disease=0, hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.shared = nn.Sequential(nn.Linear(encoder.embed_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.anomaly = nn.Linear(hidden, num_anomaly)
        self.disease = nn.Linear(hidden, num_disease) if num_disease > 0 else None

    def forward(self, waveforms):
        h = self.shared(self.encoder(waveforms)["clip_embedding"])
        out = {"anomaly": self.anomaly(h)}
        if self.disease is not None: out["disease"] = self.disease(h)
        return out
