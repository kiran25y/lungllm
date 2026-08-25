# REPO PATH: src/lungllm/eval/moe_activation_map.py
"""Phase 3 — render the per-case Expert-Activation Map from SparseMoE gate weights."""
from __future__ import annotations

EXPERT_NAMES = ["wheeze", "crackle", "both", "normal", "severity"]


def summarize(gates):
    """gates: [N, num_experts] (numpy/torch). Returns list of {expert: weight} dicts."""
    try:
        g = gates.detach().cpu().numpy()
    except Exception:
        g = gates
    return [{EXPERT_NAMES[i]: float(row[i]) for i in range(len(row))} for row in g]


def render(gates, out_path="outputs/figures/expert_activation_map.png"):
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    g = gates.detach().cpu().numpy() if hasattr(gates, "detach") else np.asarray(gates)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, max(2, g.shape[0] * 0.3)))
    plt.imshow(g, aspect="auto", cmap="viridis")
    plt.xticks(range(len(EXPERT_NAMES)), EXPERT_NAMES, rotation=45, ha="right")
    plt.ylabel("sample"); plt.colorbar(label="gate weight"); plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    return out_path
