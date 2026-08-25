# REPO PATH: src/lungllm/agent/orchestrator.py
"""Phase 8 — inference-time clinical agent (scaffold).

Closed loop (RESP-AGENT-inspired, but inference-time + retrieval):
  audio (+symptoms) -> classify -> read confidence + Expert-Activation Map
  -> retrieve similar cases (RAG) -> generate grounded report
  -> if low confidence, flag for review / request another view.
Wire `model` (the multimodal model) and `retriever` (FaissRetriever) to activate.
"""
from __future__ import annotations


class RespiratoryAgent:
    def __init__(self, model=None, retriever=None, conf_threshold=0.6):
        self.model = model
        self.retriever = retriever
        self.conf_threshold = conf_threshold

    def run(self, waveform, symptoms_text=""):
        """Returns a structured result dict; steps degrade gracefully if parts are absent."""
        result = {"steps": []}
        if self.model is None:
            result["error"] = "no model wired"; return result
        out = self.model([waveform], text=[symptoms_text] if symptoms_text else None)
        result["steps"].append("classify")
        result["anomaly"] = out.get("anomaly")
        result["expert_map"] = out.get("moe_info")
        conf = out.get("confidence", 1.0)
        if self.retriever is not None:
            q = symptoms_text or "respiratory auscultation"
            result["retrieved"] = self.retriever.retrieve(q)
            result["steps"].append("retrieve")
        result["report"] = out.get("report")
        result["needs_review"] = bool(conf < self.conf_threshold)
        result["steps"].append("report")
        return result
