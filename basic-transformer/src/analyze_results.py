import os
import torch
import json
import numpy as np

def load_metrics(log_dir, mode):
    """Load training metrics from compare_pro results"""
    path_pt = os.path.join(log_dir, f"{mode}_logs", "metrics.pt")
    path_txt = os.path.join(log_dir, f"{mode}_logs", "training_log.txt")

    data = torch.load(path_pt, map_location="cpu")
    losses = np.array(data["losses"])
    grads = np.array(data["grad_norms"])
    ents = np.array(data.get("attn_entropies", []))
    bias_stats = data.get("bias_stats", [])

    last_bias = bias_stats[-1] if bias_stats else {"mean": None, "std": None, "range": None}
    summary = {
        "final_loss": float(losses[-1]) if len(losses) else None,
        "avg_grad": float(grads.mean()) if len(grads) else None,
        "final_entropy": float(ents[-1]) if len(ents) else None,
        "bias_mean": last_bias.get("mean"),
        "bias_std": last_bias.get("std"),
        "bias_range": last_bias.get("range"),
    }

    # Optional: log file content (for reference)
    with open(path_txt, "r", encoding="utf-8") as f:
        log_text = f.read()

    return summary, log_text


def compare_metrics(add, mul):
    """Compute delta (multiplicative - additive)"""
    result = {}
    for key in add.keys():
        if add[key] is not None and mul[key] is not None:
            result[key] = round(mul[key] - add[key], 5)
        else:
            result[key] = None
    return result


def analyze_results(log_dir="logs/compare_pro"):
    modes = ["additive", "multiplicative"]

    if not all(os.path.exists(os.path.join(log_dir, f"{m}_logs")) for m in modes):
        print("❌ Missing log directories. Run compare_modes_plus_pro first.")
        return

    add, add_txt = load_metrics(log_dir, "additive")
    mul, mul_txt = load_metrics(log_dir, "multiplicative")
    diff = compare_metrics(add, mul)

    # === Numerical Report ===
    print("\n📊 ASCender Comparative Summary")
    print("─────────────────────────────────────────────")
    print(f"Additive Final Loss:        {add['final_loss']:.4f}")
    print(f"Multiplicative Final Loss:  {mul['final_loss']:.4f}")
    print(f"→ ΔLoss = {diff['final_loss']:+.4f} ({'Improved ✅' if diff['final_loss'] < 0 else 'Worse ⚠️'})")

    print(f"\nAdditive Entropy:           {add['final_entropy']:.5f}")
    print(f"Multiplicative Entropy:     {mul['final_entropy']:.5f}")

    if add["final_entropy"] and abs(add["final_entropy"]) > 1e-9:
        delta_ent = (diff['final_entropy'] / add['final_entropy']) * 100
        trend = 'Sharper Focus ✅' if diff['final_entropy'] < 0 else 'More Diffuse ⚠️'
        print(f"→ ΔEntropy = {diff['final_entropy']:+.5f} ({delta_ent:+.1f}%) {trend}")
    else:
        print("→ ΔEntropy: n/a (baseline entropy ≈ 0)")


    if add["bias_mean"] is not None:
        print(f"\nBias Mean Δ:   {diff['bias_mean']:+.5f}")
        print(f"Bias Std Δ:    {diff['bias_std']:+.5f}")
        print(f"Bias Range Δ:  {diff['bias_range']:+.5f}")

    print(f"\nGradient Norm (avg):")
    print(f"Additive={add['avg_grad']:.3f}  vs  Multiplicative={mul['avg_grad']:.3f}")
    print(f"→ ΔGrad = {diff['avg_grad']:+.5f}")
    print("─────────────────────────────────────────────")

    # === Save as JSON ===
    report = {"additive": add, "multiplicative": mul, "diff": diff}
    save_path = os.path.join(log_dir, "summary_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Saved → {save_path}")

    # === Optional: loss/entropy qualitative statement ===
    if diff['final_loss'] < 0 and diff['final_entropy'] < 0:
        print("\n✅ Conclusion: Multiplicative ASCender achieved faster convergence "
              "and sharper attention focus compared to additive baseline.")
    elif diff['final_loss'] < 0:
        print("\n⚙️ Partial Improvement: Faster convergence observed, but attention sharpness similar.")
    elif diff['final_entropy'] < 0:
        print("\n⚙️ Partial Improvement: More focused attention, but convergence unchanged.")
    else:
        print("\n⚠️ No clear improvement detected. Consider tuning beta_align / beta_coh.")

if __name__ == "__main__":
    analyze_results("logs/compare_pro")
