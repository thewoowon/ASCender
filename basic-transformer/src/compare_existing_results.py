import os
import torch
import numpy as np

def load_metrics(path):
    if not os.path.exists(path):
        print(f"⚠️ Missing: {path}")
        return None
    data = torch.load(path, map_location="cpu")
    return {
        "loss": np.mean(data.get("losses", [0])),
        "entropy": np.mean(data.get("attn_entropies", [0])),
        "grad": np.mean(data.get("grad_norms", [0])),
        "bias_mean": np.mean([b["mean"] for b in data.get("bias_stats", [])]) if data.get("bias_stats") else 0,
        "bias_std": np.mean([b["std"] for b in data.get("bias_stats", [])]) if data.get("bias_stats") else 0,
        "bias_range": np.mean([b["range"] for b in data.get("bias_stats", [])]) if data.get("bias_stats") else 0,
    }

def compare(a_name, b_name, a_path, b_path):
    a = load_metrics(a_path)
    b = load_metrics(b_path)
    if a is None or b is None:
        return

    print(f"\n📊 {a_name} → {b_name}")
    print("────────────────────────────")
    print(f"Loss:       {a['loss']:.4f} → {b['loss']:.4f}   Δ = {b['loss']-a['loss']:+.4f}")
    print(f"Entropy:    {a['entropy']:.4f} → {b['entropy']:.4f}   Δ = {b['entropy']-a['entropy']:+.4f}")
    print(f"Grad Norm:  {a['grad']:.4f} → {b['grad']:.4f}   Δ = {b['grad']-a['grad']:+.4f}")
    print(f"Bias mean:  {a['bias_mean']:.5f} → {b['bias_mean']:.5f}")
    print(f"Bias std:   {a['bias_std']:.5f} → {b['bias_std']:.5f}")
    print(f"Bias range: {a['bias_range']:.5f} → {b['bias_range']:.5f}")
    print("────────────────────────────")

if __name__ == "__main__":
    base_dir = "logs/compare_pro"
    compare("Baseline", "Additive",
            "logs/baseline_logs/metrics.pt",
            f"{base_dir}/additive_logs/metrics.pt")
    compare("Baseline", "Multiplicative",
            "logs/baseline_logs/metrics.pt",
            f"{base_dir}/multiplicative_logs/metrics.pt")
