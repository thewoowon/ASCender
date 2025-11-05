#!/usr/bin/env python3
"""
Visualize the ACTUAL bias being generated with current config settings.
Shows what patterns we're creating and suggests improvements.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ns(d, defaults=None):
    d = {} if d is None else d
    if defaults:
        for k, v in defaults.items():
            d.setdefault(k, v)
    return SimpleNamespace(**d)

def main():
    config_path = "configs/ascender256_residual.yaml"
    print(f"Loading config: {config_path}\n")
    raw = load_config(config_path)

    cfg = ns(raw)
    cfg.dataset = ns(getattr(cfg, "dataset", None))
    cfg.experiment = ns(getattr(cfg, "experiment", None))
    cfg.model = ns(getattr(cfg, "model", None))
    cfg.model.asc_cfg = ns(getattr(cfg.model, "asc_cfg", None))
    cfg.mode = getattr(cfg, "mode", "additive")

    # Load model
    if cfg.mode == "additive":
        m = importlib.import_module("src.models.transformer")
    else:
        m = importlib.import_module("src.models.multiplicative_transformer")

    Transformer, TransformerConfig = m.Transformer, m.TransformerConfig

    from src.models.ascender_bias import AscenderBiasConfig
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg.model.asc_cfg))
    if hasattr(asc_cfg_obj, "coerce"):
        asc_cfg_obj.coerce()

    model_kwargs = vars(cfg.model).copy()
    model_kwargs["asc_cfg"] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = Transformer(model_cfg).to(device)
    model.eval()

    # Find active biaser
    print("Looking for active biaser...")
    first_layer = None
    biaser = None
    layer_name = "Unknown"

    # Try decoder first
    if hasattr(model, "decoder") and len(model.decoder.layers) > 0:
        first_layer = model.decoder.layers[0]
        biaser = getattr(first_layer, "biaser_self", None)
        if biaser is None:
            biaser = getattr(first_layer.self_attn, "biaser", None)
        if biaser:
            layer_name = "Decoder L0"

    # Try encoder if decoder has no bias
    if biaser is None and hasattr(model, "encoder") and len(model.encoder.layers) > 0:
        first_layer = model.encoder.layers[0]
        biaser = getattr(first_layer.self_attn, "biaser", None)
        if biaser:
            layer_name = "Encoder L0"

    if biaser is None:
        print("❌ No active biaser found!")
        return

    print(f"✅ Found biaser in {layer_name}")
    print(f"   Type: {type(biaser).__name__}\n")

    # Print current config
    print("="*80)
    print("CURRENT BIAS CONFIGURATION")
    print("="*80)
    asc = cfg.model.asc_cfg
    print(f"use_cohesion:   {getattr(asc, 'use_cohesion', False)}")
    print(f"use_separation: {getattr(asc, 'use_separation', False)}")
    print(f"use_alignment:  {getattr(asc, 'use_alignment', False)}")
    print(f"w_coh:          {getattr(asc, 'w_coh', 0)}")
    print(f"w_sep:          {getattr(asc, 'w_sep', 0)}")
    print(f"w_align:        {getattr(asc, 'w_align', 0)}")
    print(f"sigma_coh:      {getattr(asc, 'sigma_coh', 0)}")
    print(f"sigma_sep:      {getattr(asc, 'sigma_sep', 0)}")
    print(f"global_scale:   {getattr(asc, 'global_scale_init', 1.0)}")
    print("="*80 + "\n")

    # Generate bias with different sequence lengths
    seq_lens = [32, 64, 128, 256]

    for T in seq_lens:
        print(f"\n{'='*80}")
        print(f"SEQUENCE LENGTH: {T}")
        print('='*80)

        h = torch.randn((1, T, model_cfg.d_model), device=device, dtype=torch.float32) * 0.01
        qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
        kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))

        try:
            bias_full = biaser(qh, kh, pre_q=h, pre_k=h)
        except:
            try:
                bias_full = biaser(qh, kh)
            except:
                bias_full = biaser(h, h)

        # Extract first head
        if bias_full.dim() == 4:
            bias = bias_full[0, 0].detach().cpu().numpy()
        elif bias_full.dim() == 3:
            bias = bias_full[0].detach().cpu().numpy()
        else:
            bias = bias_full.detach().cpu().numpy()

        mean = bias.mean()
        std = bias.std()
        min_val = bias.min()
        max_val = bias.max()

        print(f"\nBias Matrix Statistics:")
        print(f"  Mean:  {mean:.6f}")
        print(f"  Std:   {std:.6f}")
        print(f"  Min:   {min_val:.6f}")
        print(f"  Max:   {max_val:.6f}")
        print(f"  Range: {max_val - min_val:.6f}")

        # Check diagonal vs off-diagonal
        diag_indices = np.arange(T)
        diag_vals = bias[diag_indices, diag_indices]
        off_diag_vals = bias[~np.eye(T, dtype=bool)]

        print(f"\nDiagonal vs Off-diagonal:")
        print(f"  Diagonal mean:     {diag_vals.mean():.6f}")
        print(f"  Off-diagonal mean: {off_diag_vals.mean():.6f}")
        print(f"  Difference:        {diag_vals.mean() - off_diag_vals.mean():.6f}")

        # Visualize
        fig = plt.figure(figsize=(18, 5))

        # Panel 1: Raw bias
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(bias, cmap='coolwarm', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Bias Value')
        ax1.set_title(f"Raw Bias (T={T})\nμ={mean:.3f}, σ={std:.3f}")
        ax1.set_xlabel("Key Position")
        ax1.set_ylabel("Query Position")

        # Panel 2: Z-score normalized
        ax2 = plt.subplot(1, 3, 2)
        if std > 1e-6:
            bias_norm = (bias - mean) / std
            im2 = ax2.imshow(bias_norm, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
            plt.colorbar(im2, ax=ax2, label='Z-score')
            ax2.set_title(f"Normalized (±3σ)\nRange: [{bias_norm.min():.2f}, {bias_norm.max():.2f}]")
        else:
            im2 = ax2.imshow(bias, cmap='gray', aspect='auto')
            plt.colorbar(im2, ax=ax2)
            ax2.set_title("CONSTANT (std≈0)")
        ax2.set_xlabel("Key Position")
        ax2.set_ylabel("Query Position")

        # Panel 3: Histogram
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(bias.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.3f}')
        ax3.axvline(mean + std, color='orange', linestyle='--', linewidth=1, label=f'±1σ')
        ax3.axvline(mean - std, color='orange', linestyle='--', linewidth=1)
        ax3.set_xlabel("Bias Value")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution")
        ax3.legend()
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"logs/bias_analysis_T{T}.png", dpi=200)
        plt.close()
        print(f"\n✅ Saved: logs/bias_analysis_T{T}.png")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    sigma_coh = getattr(asc, 'sigma_coh', 50.0)
    if sigma_coh > 20:
        print(f"\n⚠️  sigma_coh={sigma_coh} is TOO LARGE for seq_len=256!")
        print(f"   This creates a very flat, wide cohesion kernel.")
        print(f"   Recommended: sigma_coh = 8.0 ~ 12.0")
        print(f"   This gives a clear diagonal band pattern.")

    w_coh = getattr(asc, 'w_coh', 5.0)
    if std < 0.5:
        print(f"\n⚠️  Bias std={std:.3f} is too small!")
        print(f"   The pattern is too weak to be visible.")
        print(f"   Consider increasing:")
        print(f"   - w_coh to 10.0 ~ 15.0 (currently {w_coh})")
        print(f"   - global_scale_init to 2.0 ~ 3.0")

    print("\n✅ Analysis complete! Check logs/bias_analysis_T*.png files.")
    print("="*80)

if __name__ == "__main__":
    main()
