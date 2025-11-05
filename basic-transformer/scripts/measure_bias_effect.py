#!/usr/bin/env python3
"""
Diagnostic script to measure ASCender bias effect.

This script:
1. Loads a trained model
2. Runs inference with bias ON vs OFF
3. Computes multiple metrics to quantify bias contribution
4. Generates visualizations

Usage:
    python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml --checkpoint path/to/checkpoint.pt
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from types import SimpleNamespace

# Assume imports work (adjust paths as needed)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import Transformer, TransformerConfig
from src.models.ascender_bias import AscenderBiasConfig


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ns(d, defaults=None):
    d = {} if d is None else d
    if defaults:
        for k, v in defaults.items():
            d.setdefault(k, v)
    return SimpleNamespace(**d)


@torch.no_grad()
def measure_bias_contribution(model, batch, device):
    """
    Quantify how much bias affects model predictions.

    Returns dict with metrics:
    - logit_diff: Mean absolute difference in logits
    - prob_diff: Mean absolute difference in probabilities
    - kl_div: KL divergence between output distributions
    - top1_agreement: Fraction where top-1 prediction matches
    - entropy_change: How bias affects output entropy
    - attention_diff: How much attention patterns change
    """
    model.eval()
    src, tgt_inp, tgt_out = (x.to(device) for x in batch)

    # Forward with bias ON
    logits_on = model(src, tgt_inp)
    probs_on = F.softmax(logits_on, dim=-1)

    # Capture attention patterns with bias
    attn_on = []
    for layer in model.decoder.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "last_attn"):
            attn_on.append(layer.self_attn.last_attn.clone())

    # Turn off bias
    saved_biasers = []
    for layer in model.decoder.layers:
        sa_biaser = getattr(layer.self_attn, "biaser", None)
        ca_biaser = getattr(layer.cross_attn, "biaser", None) if hasattr(layer, "cross_attn") else None
        saved_biasers.append((sa_biaser, ca_biaser))

        layer.self_attn.biaser = None
        if hasattr(layer, "cross_attn"):
            layer.cross_attn.biaser = None

    # Forward with bias OFF
    logits_off = model(src, tgt_inp)
    probs_off = F.softmax(logits_off, dim=-1)

    # Capture attention patterns without bias
    attn_off = []
    for layer in model.decoder.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "last_attn"):
            attn_off.append(layer.self_attn.last_attn.clone())

    # Restore biasers
    for layer, (sa_b, ca_b) in zip(model.decoder.layers, saved_biasers):
        layer.self_attn.biaser = sa_b
        if hasattr(layer, "cross_attn"):
            layer.cross_attn.biaser = ca_b

    # Compute metrics
    metrics = {}

    # Logit differences
    logit_diff = torch.abs(logits_on - logits_off)
    metrics["logit_diff_mean"] = float(logit_diff.mean().item())
    metrics["logit_diff_max"] = float(logit_diff.max().item())
    metrics["logit_diff_std"] = float(logit_diff.std().item())

    # Probability differences
    prob_diff = torch.abs(probs_on - probs_off)
    metrics["prob_diff_mean"] = float(prob_diff.mean().item())
    metrics["prob_diff_max"] = float(prob_diff.max().item())

    # KL divergence (ON vs OFF)
    eps = 1e-8
    kl_on_off = (probs_on * (torch.log(probs_on + eps) - torch.log(probs_off + eps))).sum(dim=-1)
    metrics["kl_on_off_mean"] = float(kl_on_off.mean().item())
    metrics["kl_on_off_max"] = float(kl_on_off.max().item())

    # Top-1 prediction agreement
    top1_on = logits_on.argmax(dim=-1)
    top1_off = logits_off.argmax(dim=-1)
    agreement = (top1_on == top1_off).float().mean()
    metrics["top1_agreement"] = float(agreement.item())
    metrics["top1_disagreement"] = float((1.0 - agreement).item())

    # Entropy change
    entropy_on = -(probs_on * torch.log(probs_on + eps)).sum(dim=-1).mean()
    entropy_off = -(probs_off * torch.log(probs_off + eps)).sum(dim=-1).mean()
    metrics["entropy_on"] = float(entropy_on.item())
    metrics["entropy_off"] = float(entropy_off.item())
    metrics["entropy_change"] = float((entropy_on - entropy_off).item())

    # Attention pattern differences
    if attn_on and attn_off:
        attn_diffs = []
        for a_on, a_off in zip(attn_on, attn_off):
            diff = torch.abs(a_on - a_off).mean()
            attn_diffs.append(float(diff.item()))
        metrics["attention_diff_mean"] = float(np.mean(attn_diffs))
        metrics["attention_diff_max"] = float(np.max(attn_diffs))
        metrics["attention_diff_per_layer"] = attn_diffs

    # Loss comparison (on target tokens)
    pad_id = model.cfg.pad_id
    valid_mask = (tgt_out != pad_id)
    nll_on = F.cross_entropy(logits_on.view(-1, logits_on.size(-1)), tgt_out.view(-1),
                               ignore_index=pad_id, reduction='mean')
    nll_off = F.cross_entropy(logits_off.view(-1, logits_off.size(-1)), tgt_out.view(-1),
                                ignore_index=pad_id, reduction='mean')
    metrics["nll_on"] = float(nll_on.item())
    metrics["nll_off"] = float(nll_off.item())
    metrics["nll_diff"] = float((nll_on - nll_off).item())
    metrics["nll_improvement"] = float((nll_off - nll_on).item())  # positive = bias helps

    return metrics


@torch.no_grad()
def visualize_bias_magnitude(model, device, save_path="bias_magnitude.png"):
    """
    Generate sample bias matrices and plot their magnitude distribution.
    """
    model.eval()

    # Generate a dummy input
    T = 32
    dummy_input = torch.randn(1, T, model.cfg.d_model, device=device) * 0.01

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for li in range(min(2, len(model.decoder.layers))):
        layer = model.decoder.layers[li]
        biaser = getattr(layer, "biaser_self", None)
        if biaser is None:
            continue

        # Generate bias
        mha = layer.self_attn
        qh = mha._shape(mha.q_proj(dummy_input))
        kh = mha._shape(mha.k_proj(dummy_input))
        bias = biaser(qh, kh, pre_q=dummy_input, pre_k=dummy_input)  # (1, H, T, T)

        # Plot heatmap (head 0)
        ax = axes[li, 0]
        bias_h0 = bias[0, 0].cpu().numpy()
        im = ax.imshow(bias_h0, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"Layer {li} Head 0 Bias")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.colorbar(im, ax=ax)

        # Plot distribution across all heads
        ax = axes[li, 1]
        bias_flat = bias[0].flatten().cpu().numpy()
        ax.hist(bias_flat, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"Layer {li} Bias Distribution")
        ax.set_xlabel("Bias Value")
        ax.set_ylabel("Frequency")
        ax.text(0.05, 0.95, f"Mean: {bias_flat.mean():.3f}\nStd: {bias_flat.std():.3f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved bias magnitude visualization to {save_path}")
    plt.close()


def print_metrics(metrics):
    """Pretty print all metrics."""
    print("\n" + "="*60)
    print("BIAS CONTRIBUTION METRICS")
    print("="*60)

    print("\n📊 Output Differences:")
    print(f"  Logit Δ (mean):     {metrics['logit_diff_mean']:.4f}")
    print(f"  Logit Δ (max):      {metrics['logit_diff_max']:.4f}")
    print(f"  Prob Δ (mean):      {metrics['prob_diff_mean']:.4f}")
    print(f"  KL divergence:      {metrics['kl_on_off_mean']:.4f}")

    print("\n🎯 Prediction Changes:")
    print(f"  Top-1 agreement:    {metrics['top1_agreement']*100:.1f}%")
    print(f"  Top-1 disagreement: {metrics['top1_disagreement']*100:.1f}%")

    print("\n📉 Loss Impact:")
    print(f"  NLL (bias ON):      {metrics['nll_on']:.4f}")
    print(f"  NLL (bias OFF):     {metrics['nll_off']:.4f}")
    print(f"  NLL improvement:    {metrics['nll_improvement']:.4f} {'✅' if metrics['nll_improvement'] > 0 else '❌'}")

    print("\n🔍 Attention Changes:")
    print(f"  Attention Δ (mean): {metrics.get('attention_diff_mean', 0):.4f}")
    if 'attention_diff_per_layer' in metrics:
        for i, diff in enumerate(metrics['attention_diff_per_layer']):
            print(f"    Layer {i}:         {diff:.4f}")

    print("\n💡 Interpretation:")
    if metrics['top1_disagreement'] < 0.01:
        print("  ⚠️  VERY LOW EFFECT - Bias changes <1% of predictions")
        print("      Recommendation: Use more aggressive config")
    elif metrics['top1_disagreement'] < 0.05:
        print("  ⚠️  LOW EFFECT - Bias changes <5% of predictions")
        print("      Recommendation: Increase std_match_ratio or component weights")
    elif metrics['top1_disagreement'] < 0.15:
        print("  ✅ MODERATE EFFECT - Bias meaningfully affects predictions")
    else:
        print("  ⚠️  HIGH EFFECT - Bias changes >15% of predictions")
        print("      This might be too aggressive if NLL worsens")

    if metrics['nll_improvement'] > 0.05:
        print("  ✅ HELPFUL - Bias significantly improves loss")
    elif metrics['nll_improvement'] > 0:
        print("  ✅ SLIGHTLY HELPFUL - Small improvement")
    elif abs(metrics['nll_improvement']) < 0.01:
        print("  ⚠️  NEUTRAL - No clear improvement or harm")
    else:
        print("  ❌ HARMFUL - Bias increases loss")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=10)
    args = parser.parse_args()

    # Load config
    raw = load_config(args.config)
    cfg = ns(raw)
    cfg.dataset = ns(getattr(cfg, "dataset", None))
    cfg.experiment = ns(getattr(cfg, "experiment", None))
    cfg.model = ns(getattr(cfg, "model", None))
    cfg.model.asc_cfg = ns(getattr(cfg.model, "asc_cfg", None))

    # Build model config
    from src.models.ascender_bias import AscenderBiasConfig
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg.model.asc_cfg))
    if hasattr(asc_cfg_obj, "coerce"):
        asc_cfg_obj.coerce()

    model_kwargs = vars(cfg.model).copy()
    model_kwargs["asc_cfg"] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = Transformer(model_cfg).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("⚠️  No checkpoint provided - using randomly initialized model")

    # Generate dummy data
    print(f"\nGenerating {args.num_batches} batches of dummy data...")
    vocab_size = model_cfg.src_vocab_size
    pad_id = model_cfg.pad_id

    all_metrics = []
    for i in range(args.num_batches):
        src = torch.randint(1, vocab_size, (args.batch_size, args.seq_len), device=device)
        tgt_inp = torch.randint(1, vocab_size, (args.batch_size, args.seq_len), device=device)
        tgt_out = torch.randint(1, vocab_size, (args.batch_size, args.seq_len), device=device)

        batch = (src, tgt_inp, tgt_out)
        metrics = measure_bias_contribution(model, batch, device)
        all_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == "attention_diff_per_layer":
            continue  # Skip nested list
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)

    if "attention_diff_per_layer" in all_metrics[0]:
        num_layers = len(all_metrics[0]["attention_diff_per_layer"])
        avg_metrics["attention_diff_per_layer"] = [
            np.mean([m["attention_diff_per_layer"][i] for m in all_metrics])
            for i in range(num_layers)
        ]

    # Print results
    print_metrics(avg_metrics)

    # Visualize
    visualize_bias_magnitude(model, device, save_path="logs/bias_magnitude.png")


if __name__ == "__main__":
    main()
