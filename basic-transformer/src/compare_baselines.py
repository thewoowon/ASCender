# src/compare_baselines.py
import json, numpy as np
from pathlib import Path

def load_results(dirpath):
    losses, entropies = [], []
    for f in Path(dirpath).glob("results_seed*.json"):
        data = json.load(open(f))
        losses.append(data["final_loss"])
        entropies.append(data["entropy"])
    return np.mean(losses), np.std(losses), np.mean(entropies)

def compare(a_name, b_name, a_dir, b_dir):
    a_loss, a_std, a_ent = load_results(a_dir)
    b_loss, b_std, b_ent = load_results(b_dir)
    print(f"\n📊 {a_name} vs {b_name}")
    print("──────────────────────────────")
    print(f"{a_name:<12} Loss: {a_loss:.4f} ± {a_std:.4f}")
    print(f"{b_name:<12} Loss: {b_loss:.4f} ± {a_std:.4f}")
    print(f"ΔLoss = {b_loss - a_loss:+.4f}")
    print(f"{a_name:<12} Entropy: {a_ent:.4f}")
    print(f"{b_name:<12} Entropy: {b_ent:.4f}")
    print(f"ΔEntropy = {b_ent - a_ent:+.4f}")
    print("──────────────────────────────")

if __name__ == "__main__":
    compare("Baseline", "Additive",
            "logs/baseline", "logs/additive")
    compare("Baseline", "Multiplicative",
            "logs/baseline", "logs/multiplicative")
