#!/usr/bin/env python3
"""
Extract and analyze learned alpha mixing weights from trained ASCender model.
Shows how much each head chose to use spatial bias vs. learned attention.

Usage:
    python extract_alpha.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.transformer import Transformer, TransformerConfig
from src.models.ascender_bias import AscenderBiasConfig
import yaml


def load_model_from_config(config_path: str):
    """Load model architecture from config (for inspection)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Build config
    from types import SimpleNamespace
    def ns(d):
        return SimpleNamespace(**d) if d else SimpleNamespace()

    cfg = ns(raw)
    cfg.dataset = ns(raw.get('dataset'))
    cfg.experiment = ns(raw.get('experiment'))
    cfg.model = ns(raw.get('model'))
    cfg.model.asc_cfg = ns(raw['model'].get('asc_cfg'))

    # Build ASCender config
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg.model.asc_cfg))
    if hasattr(asc_cfg_obj, 'coerce'):
        asc_cfg_obj.coerce()

    model_kwargs = vars(cfg.model).copy()
    model_kwargs['asc_cfg'] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    model = Transformer(model_cfg)
    return model, model_cfg


def extract_alpha_values(model):
    """Extract alpha values from all attention layers."""
    results = {
        'encoder': [],
        'decoder': []
    }

    # Encoder layers
    for i, layer in enumerate(model.encoder.layers):
        if hasattr(layer.self_attn, 'alpha_logit'):
            alpha_logit = layer.self_attn.alpha_logit.detach()
            alpha = torch.sigmoid(alpha_logit)  # Convert logit to [0,1]

            results['encoder'].append({
                'layer': i,
                'alpha_logit': alpha_logit.cpu().numpy(),
                'alpha_effective': alpha.cpu().numpy(),
                'has_biaser': layer.self_attn.biaser is not None,
                'residual_path': getattr(layer.self_attn, 'enable_residual_path', False)
            })

    # Decoder layers
    for i, layer in enumerate(model.decoder.layers):
        # Self-attention
        if hasattr(layer.self_attn, 'alpha_logit'):
            alpha_logit = layer.self_attn.alpha_logit.detach()
            alpha = torch.sigmoid(alpha_logit)

            results['decoder'].append({
                'layer': i,
                'type': 'self',
                'alpha_logit': alpha_logit.cpu().numpy(),
                'alpha_effective': alpha.cpu().numpy(),
                'has_biaser': layer.self_attn.biaser is not None,
                'residual_path': getattr(layer.self_attn, 'enable_residual_path', False)
            })

        # Cross-attention
        if hasattr(layer.cross_attn, 'alpha_logit'):
            alpha_logit = layer.cross_attn.alpha_logit.detach()
            alpha = torch.sigmoid(alpha_logit)

            results['decoder'].append({
                'layer': i,
                'type': 'cross',
                'alpha_logit': alpha_logit.cpu().numpy(),
                'alpha_effective': alpha.cpu().numpy(),
                'has_biaser': layer.cross_attn.biaser is not None,
                'residual_path': getattr(layer.cross_attn, 'enable_residual_path', False)
            })

    return results


def print_analysis(results):
    """Print detailed analysis of alpha values."""

    print("=" * 80)
    print("ALPHA MIXING WEIGHT ANALYSIS")
    print("=" * 80)
    print()
    print("α (alpha) interpretation:")
    print("  α = 1.0 → 100% learned attention, 0% spatial bias (bias IGNORED)")
    print("  α = 0.5 → 50/50 mix (bias has moderate influence)")
    print("  α = 0.0 → 0% learned attention, 100% spatial bias (bias DOMINATES)")
    print()
    print("=" * 80)

    # Encoder analysis
    if results['encoder']:
        print("\n🔵 ENCODER (Bidirectional - Symmetric Neighborhoods)")
        print("-" * 80)

        for info in results['encoder']:
            layer = info['layer']
            alpha = info['alpha_effective']
            has_bias = "✅ HAS BIAS" if info['has_biaser'] else "❌ NO BIAS"
            residual = "✅ RESIDUAL PATH" if info['residual_path'] else "❌ STANDARD"

            print(f"\nLayer {layer} | {has_bias} | {residual}")
            print(f"  Alpha per head: {alpha}")
            print(f"  Mean:   {alpha.mean():.4f}")
            print(f"  Median: {float(sorted(alpha)[len(alpha)//2]):.4f}")
            print(f"  Min:    {alpha.min():.4f}")
            print(f"  Max:    {alpha.max():.4f}")
            print(f"  Std:    {alpha.std():.4f}")

            # Interpretation
            mean_alpha = alpha.mean()
            if mean_alpha > 0.9:
                print(f"  ⚠️  INTERPRETATION: Model learned to IGNORE spatial bias (α≈1.0)")
            elif mean_alpha > 0.7:
                print(f"  📊 INTERPRETATION: Model mostly uses learned attention (bias weak)")
            elif mean_alpha > 0.3:
                print(f"  ⚖️  INTERPRETATION: Balanced mix (bias has moderate influence)")
            elif mean_alpha > 0.1:
                print(f"  📈 INTERPRETATION: Model mostly uses spatial bias (bias strong)")
            else:
                print(f"  🎯 INTERPRETATION: Model relies on spatial bias (α≈0.0)")

    # Decoder analysis
    if results['decoder']:
        print("\n\n🔴 DECODER (Causal - Asymmetric Neighborhoods)")
        print("-" * 80)

        for info in results['decoder']:
            layer = info['layer']
            attn_type = info['type']
            alpha = info['alpha_effective']
            has_bias = "✅ HAS BIAS" if info['has_biaser'] else "❌ NO BIAS"
            residual = "✅ RESIDUAL PATH" if info['residual_path'] else "❌ STANDARD"

            print(f"\nLayer {layer} [{attn_type}-attn] | {has_bias} | {residual}")
            print(f"  Alpha per head: {alpha}")
            print(f"  Mean:   {alpha.mean():.4f}")
            print(f"  Median: {float(sorted(alpha)[len(alpha)//2]):.4f}")
            print(f"  Min:    {alpha.min():.4f}")
            print(f"  Max:    {alpha.max():.4f}")
            print(f"  Std:    {alpha.std():.4f}")

            # Interpretation
            mean_alpha = alpha.mean()
            if mean_alpha > 0.9:
                print(f"  ⚠️  INTERPRETATION: Model learned to IGNORE spatial bias (α≈1.0)")
            elif mean_alpha > 0.7:
                print(f"  📊 INTERPRETATION: Model mostly uses learned attention (bias weak)")
            elif mean_alpha > 0.3:
                print(f"  ⚖️  INTERPRETATION: Balanced mix (bias has moderate influence)")
            elif mean_alpha > 0.1:
                print(f"  📈 INTERPRETATION: Model mostly uses spatial bias (bias strong)")
            else:
                print(f"  🎯 INTERPRETATION: Model relies on spatial bias (α≈0.0)")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    all_alphas = []
    for enc_info in results['encoder']:
        if enc_info['has_biaser']:
            all_alphas.extend(enc_info['alpha_effective'].tolist())
    for dec_info in results['decoder']:
        if dec_info['has_biaser']:
            all_alphas.extend(dec_info['alpha_effective'].tolist())

    if all_alphas:
        import numpy as np
        all_alphas = np.array(all_alphas)

        print(f"\nAcross ALL heads with spatial bias:")
        print(f"  Mean α:   {all_alphas.mean():.4f}")
        print(f"  Median α: {np.median(all_alphas):.4f}")
        print(f"  Std α:    {all_alphas.std():.4f}")

        # Final verdict
        mean_alpha = all_alphas.mean()
        print("\n" + "=" * 80)
        print("🎯 FINAL VERDICT:")
        print("=" * 80)

        if mean_alpha > 0.95:
            print("❌ Spatial bias COMPLETELY IGNORED by model (α > 0.95)")
            print("   Model learned spatial structure provides ZERO value.")
            print("   Conclusion: Boids-inspired biases don't help language modeling.")
        elif mean_alpha > 0.85:
            print("⚠️  Spatial bias MOSTLY IGNORED by model (α > 0.85)")
            print("   Model learned spatial structure provides minimal value.")
        elif mean_alpha > 0.70:
            print("📊 Spatial bias has WEAK influence (α > 0.70)")
            print("   Model prefers learned attention but uses some spatial info.")
        elif mean_alpha > 0.30:
            print("⚖️  BALANCED use of spatial bias (0.30 < α < 0.70)")
            print("   Model finds value in mixing both signals.")
        elif mean_alpha > 0.15:
            print("📈 Spatial bias has STRONG influence (α < 0.30)")
            print("   Model relies heavily on spatial structure.")
        else:
            print("🎯 Spatial bias DOMINATES (α < 0.15)")
            print("   Model learned spatial structure is more valuable than content!")

        print()
    else:
        print("\n⚠️  No layers with spatial bias found!")

    print("=" * 80)


def main():
    # Try to find config
    config_path = "configs/ascender256_residual.yaml"

    if not Path(config_path).exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please specify the correct config path.")
        return

    print(f"Loading model from config: {config_path}")
    print("(Note: This shows INITIAL alpha values. For trained values, load checkpoint.)")
    print()

    model, cfg = load_model_from_config(config_path)

    # Check if residual path is enabled
    residual_enabled = getattr(cfg, 'enable_residual_path', False)
    print(f"Residual Path Enabled: {residual_enabled}")
    print(f"ASCender Enabled: {cfg.use_ascender}")
    print(f"Encoder Bias: {cfg.asc_bias_enc}")
    print(f"Decoder Self Bias: {cfg.asc_bias_dec_self}")
    print(f"Decoder Cross Bias: {cfg.asc_bias_dec_cross}")
    print()

    if not residual_enabled:
        print("⚠️  WARNING: Residual path is NOT enabled!")
        print("   Alpha values will not be used (standard additive bias).")
        print()

    # Extract alpha values
    results = extract_alpha_values(model)

    # Print analysis
    print_analysis(results)

    # Save to file
    import json
    output = {
        'encoder': [
            {
                'layer': r['layer'],
                'alpha_effective': r['alpha_effective'].tolist(),
                'has_biaser': r['has_biaser'],
                'residual_path': r['residual_path']
            }
            for r in results['encoder']
        ],
        'decoder': [
            {
                'layer': r['layer'],
                'type': r['type'],
                'alpha_effective': r['alpha_effective'].tolist(),
                'has_biaser': r['has_biaser'],
                'residual_path': r['residual_path']
            }
            for r in results['decoder']
        ]
    }

    output_path = "alpha_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Alpha values saved to: {output_path}")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. If these are INITIAL values (untrained):")
    print("   - Train your model first")
    print("   - Then modify this script to load the trained checkpoint")
    print()
    print("2. If these are TRAINED values:")
    print("   - Share the alpha values above for deeper analysis")
    print("   - We'll analyze what the model learned")
    print()
    print("3. To load from checkpoint:")
    print("   - Modify this script to:")
    print("     model.load_state_dict(torch.load('path/to/checkpoint.pt'))")
    print("=" * 80)


if __name__ == "__main__":
    main()
