#!/usr/bin/env python3
"""
Diagnose why bias heatmaps are showing uniform colors.
Check actual bias value distributions and statistics.
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
    # Load config
    config_path = "configs/ascender256_residual.yaml"
    print(f"Loading config: {config_path}")
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

    # Build config
    from src.models.ascender_bias import AscenderBiasConfig
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg.model.asc_cfg))

    if hasattr(asc_cfg_obj, "coerce"):
        asc_cfg_obj.coerce()

    model_kwargs = vars(cfg.model).copy()
    model_kwargs["asc_cfg"] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    # Create model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Transformer(model_cfg).to(device)
    model.eval()

    print("\n" + "="*80)
    print("BIAS DIAGNOSTIC REPORT")
    print("="*80)

    # Check encoder layers
    print("\n[ENCODER LAYERS]")
    for i, layer in enumerate(model.encoder.layers):
        biaser = getattr(layer.self_attn, "biaser", None)
        if biaser is None:
            print(f"  Layer {i}: NO BIASER")
            continue

        print(f"\n  Layer {i}: BIASER FOUND")
        print(f"    Type: {type(biaser).__name__}")

        # Generate sample bias
        T = 32  # sequence length
        h = torch.randn((1, T, model_cfg.d_model), device=device, dtype=torch.float32) * 0.01

        try:
            qh = layer.self_attn._shape(layer.self_attn.q_proj(h))
            kh = layer.self_attn._shape(layer.self_attn.k_proj(h))

            # Try different call signatures
            try:
                bias = biaser(qh, kh, pre_q=h, pre_k=h)
            except:
                try:
                    bias = biaser(qh, kh)
                except:
                    bias = biaser(h, h)

            if bias.dim() == 4:  # (B, H, T, T)
                bias_sample = bias[0, 0].detach().cpu().numpy()
            elif bias.dim() == 3:  # (B, T, T) or (H, T, T)
                bias_sample = bias[0].detach().cpu().numpy()
            else:
                bias_sample = bias.detach().cpu().numpy()

            mean = bias_sample.mean()
            std = bias_sample.std()
            min_val = bias_sample.min()
            max_val = bias_sample.max()

            print(f"    Shape: {bias_sample.shape}")
            print(f"    Mean:  {mean:.6f}")
            print(f"    Std:   {std:.6f}")
            print(f"    Min:   {min_val:.6f}")
            print(f"    Max:   {max_val:.6f}")
            print(f"    Range: {max_val - min_val:.6f}")

            # Check if values are essentially constant
            if std < 1e-4:
                print(f"    ⚠️  WARNING: Bias values are nearly CONSTANT (std={std:.2e})")
                print(f"        This will appear as a solid color in heatmap!")

            # Visualize histogram
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.hist(bias_sample.flatten(), bins=50, edgecolor='black')
            plt.title(f"Encoder L{i}: Bias Distribution")
            plt.xlabel("Bias Value")
            plt.ylabel("Frequency")

            plt.subplot(1, 3, 2)
            plt.imshow(bias_sample, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Raw Value')
            plt.title(f"Raw Bias (range: {max_val-min_val:.4f})")

            plt.subplot(1, 3, 3)
            if std > 1e-6:
                normalized = (bias_sample - mean) / std
                plt.imshow(normalized, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
                plt.colorbar(label='Z-score')
                plt.title(f"Normalized (σ={std:.4f})")
            else:
                plt.imshow(bias_sample, cmap='gray', aspect='auto')
                plt.colorbar(label='Value')
                plt.title("CONSTANT (std≈0)")

            plt.tight_layout()
            plt.savefig(f"logs/diagnostic_encoder_L{i}.png", dpi=150)
            plt.close()
            print(f"    Saved: logs/diagnostic_encoder_L{i}.png")

        except Exception as e:
            print(f"    ❌ ERROR generating bias: {e}")

    # Check decoder layers
    print("\n[DECODER LAYERS]")
    for i, layer in enumerate(model.decoder.layers):
        biaser = getattr(layer.self_attn, "biaser", None)
        layer_biaser = getattr(layer, "biaser_self", None)

        if biaser is None and layer_biaser is None:
            print(f"  Layer {i}: NO BIASER")
            continue

        active_biaser = biaser if biaser is not None else layer_biaser
        print(f"\n  Layer {i}: BIASER FOUND")
        print(f"    Type: {type(active_biaser).__name__}")
        print(f"    Location: {'self_attn.biaser' if biaser else 'layer.biaser_self'}")

        # Generate sample bias
        T = 32
        h = torch.randn((1, T, model_cfg.d_model), device=device, dtype=torch.float32) * 0.01

        try:
            qh = layer.self_attn._shape(layer.self_attn.q_proj(h))
            kh = layer.self_attn._shape(layer.self_attn.k_proj(h))

            try:
                bias = active_biaser(qh, kh, pre_q=h, pre_k=h)
            except:
                try:
                    bias = active_biaser(qh, kh)
                except:
                    bias = active_biaser(h, h)

            if bias.dim() == 4:
                bias_sample = bias[0, 0].detach().cpu().numpy()
            elif bias.dim() == 3:
                bias_sample = bias[0].detach().cpu().numpy()
            else:
                bias_sample = bias.detach().cpu().numpy()

            mean = bias_sample.mean()
            std = bias_sample.std()
            min_val = bias_sample.min()
            max_val = bias_sample.max()

            print(f"    Shape: {bias_sample.shape}")
            print(f"    Mean:  {mean:.6f}")
            print(f"    Std:   {std:.6f}")
            print(f"    Min:   {min_val:.6f}")
            print(f"    Max:   {max_val:.6f}")
            print(f"    Range: {max_val - min_val:.6f}")

            if std < 1e-4:
                print(f"    ⚠️  WARNING: Bias values are nearly CONSTANT (std={std:.2e})")
                print(f"        This will appear as a solid color in heatmap!")

            # Visualize
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.hist(bias_sample.flatten(), bins=50, edgecolor='black')
            plt.title(f"Decoder L{i}: Bias Distribution")
            plt.xlabel("Bias Value")
            plt.ylabel("Frequency")

            plt.subplot(1, 3, 2)
            plt.imshow(bias_sample, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Raw Value')
            plt.title(f"Raw Bias (range: {max_val-min_val:.4f})")

            plt.subplot(1, 3, 3)
            if std > 1e-6:
                normalized = (bias_sample - mean) / std
                plt.imshow(normalized, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
                plt.colorbar(label='Z-score')
                plt.title(f"Normalized (σ={std:.4f})")
            else:
                plt.imshow(bias_sample, cmap='gray', aspect='auto')
                plt.colorbar(label='Value')
                plt.title("CONSTANT (std≈0)")

            plt.tight_layout()
            plt.savefig(f"logs/diagnostic_decoder_L{i}.png", dpi=150)
            plt.close()
            print(f"    Saved: logs/diagnostic_decoder_L{i}.png")

        except Exception as e:
            print(f"    ❌ ERROR generating bias: {e}")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nCheck logs/diagnostic_*.png files to see bias distributions.")
    print("\nIf std is very small (< 1e-4), the bias is essentially constant")
    print("and will appear as a single color regardless of visualization method.")

if __name__ == "__main__":
    main()
