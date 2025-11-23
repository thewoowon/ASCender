"""
Ablation Study - ASCender v2.0

Test each component and level separately:
1. Component Ablation: A-only, S-only, C-only, A+S, A+C, S+C, A+S+C
2. Level Ablation: L1-only, L2-only, L3-only, L1+L2, L1+L3, L2+L3, All
3. α Analysis: Track α evolution across configurations

This is the KEY analysis for the paper!
"""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'src'))

import torch
import torch.nn as nn
import numpy as np
import json
from models.point_ascender_v2 import ASCenderV2Config

# Reuse model from RBP experiment
from tiny_model_with_rbp import (
    TinyPointTransformerRBP,
    generate_synthetic_dataset,
    train_epoch,
    eval_model,
)


def run_ablation_study():
    print("="*70)
    print(" "*15 + "ABLATION STUDY - ASCender v2.0")
    print("="*70)
    print("\nThis is the KEY analysis for the paper!")
    print("  1. Component Ablation (A/S/C)")
    print("  2. Level Ablation (L1/L2/L3)")
    print("  3. α Evolution Analysis")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Config
    num_classes = 10
    num_points = 128
    batch_size = 32
    epochs = 100
    lr = 0.001

    # Dataset
    print("\n📊 Generating dataset...")
    train_data = generate_synthetic_dataset(800, num_points, num_classes)
    val_data = generate_synthetic_dataset(200, num_points, num_classes, seed=43)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # ========================================================================
    # Part 1: COMPONENT ABLATION (A/S/C)
    # ========================================================================
    print("\n" + "="*70)
    print(" "*20 + "PART 1: COMPONENT ABLATION")
    print("="*70)

    component_configs = [
        ("Baseline (No ASC)", None),
        ("Alignment Only (A)", ASCenderV2Config(
            use_alignment=True, use_separation=False, use_cohesion=False,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=1.0,
        )),
        ("Separation Only (S)", ASCenderV2Config(
            use_alignment=False, use_separation=True, use_cohesion=False,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_sep=1.0,
        )),
        ("Cohesion Only (C)", ASCenderV2Config(
            use_alignment=False, use_separation=False, use_cohesion=True,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_coh=1.0,
        )),
        ("A + S", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=False,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=0.7, w_sep=0.5,
        )),
        ("A + C", ASCenderV2Config(
            use_alignment=True, use_separation=False, use_cohesion=True,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=0.7, w_coh=0.7,
        )),
        ("S + C", ASCenderV2Config(
            use_alignment=False, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_sep=0.5, w_coh=0.7,
        )),
        ("A + S + C (Full)", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False, enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
    ]

    component_results = {}

    for name, config in component_configs:
        print(f"\n🚀 Testing: {name}")

        if config is None:
            # Baseline
            model = TinyPointTransformerRBP(num_classes=num_classes, use_ascender=False).to(device)
        else:
            model = TinyPointTransformerRBP(num_classes=num_classes, use_ascender=True, asc_config=config).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        for epoch in range(epochs):
            train_epoch(model, train_loader, opt, criterion, device)

        # Eval
        _, acc = eval_model(model, val_loader, criterion, device)
        alpha = model.get_alpha() if hasattr(model, 'get_alpha') else None

        component_results[name] = {
            'accuracy': acc,
            'alpha': alpha if alpha is not None else None,
        }

        alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
        print(f"   Accuracy: {acc:.3f}, α: {alpha_str}")

    # ========================================================================
    # Part 2: LEVEL ABLATION (L1/L2/L3)
    # ========================================================================
    print("\n" + "="*70)
    print(" "*20 + "PART 2: LEVEL ABLATION")
    print("="*70)

    level_configs = [
        ("Baseline (No Levels)", None),
        ("L1: Graph Only", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=True,
            enable_vector_kernel=False,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
        ("L2: Kernel Only", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False,
            enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
        ("L3: Value Only", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False,
            enable_vector_kernel=True,  # Need B_vec for L3
            enable_value_rbp=True, enable_value_gating=True,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
        ("L1 + L2", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=True,
            enable_vector_kernel=True,
            enable_value_rbp=False, enable_value_gating=False,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
        ("L2 + L3", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False,
            enable_vector_kernel=True,
            enable_value_rbp=True, enable_value_gating=True,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
        ("L1 + L2 + L3 (Full)", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=True,
            enable_vector_kernel=True,
            enable_value_rbp=True, enable_value_gating=True,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
    ]

    level_results = {}

    for name, config in level_configs:
        print(f"\n🚀 Testing: {name}")

        if config is None:
            model = TinyPointTransformerRBP(num_classes=num_classes, use_ascender=False).to(device)
        else:
            model = TinyPointTransformerRBP(num_classes=num_classes, use_ascender=True, asc_config=config).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        for epoch in range(epochs):
            train_epoch(model, train_loader, opt, criterion, device)

        # Eval
        _, acc = eval_model(model, val_loader, criterion, device)
        alpha = model.get_alpha() if hasattr(model, 'get_alpha') else None

        level_results[name] = {
            'accuracy': acc,
            'alpha': alpha if alpha is not None else None,
        }

        alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
        print(f"   Accuracy: {acc:.3f}, α: {alpha_str}")

    # ========================================================================
    # ANALYSIS & SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print(" "*25 + "RESULTS SUMMARY")
    print("="*70)

    # Component ablation
    print("\n📊 COMPONENT ABLATION (A/S/C):")
    print("-"*70)
    baseline_acc = component_results["Baseline (No ASC)"]['accuracy']

    for name, res in component_results.items():
        delta = res['accuracy'] - baseline_acc
        alpha_str = f"{res['alpha']:.4f}" if res['alpha'] else "N/A"
        print(f"{name:20s} | Acc: {res['accuracy']:.3f} (Δ {delta:+.3f}) | α: {alpha_str}")

    # Find best component
    best_comp = max([k for k in component_results.keys() if k != "Baseline (No ASC)"],
                    key=lambda k: component_results[k]['accuracy'])
    print(f"\n✅ Best Component Config: {best_comp}")
    print(f"   Accuracy: {component_results[best_comp]['accuracy']:.3f}")

    # Level ablation
    print("\n📊 LEVEL ABLATION (L1/L2/L3):")
    print("-"*70)

    for name, res in level_results.items():
        delta = res['accuracy'] - baseline_acc
        alpha_str = f"{res['alpha']:.4f}" if res['alpha'] else "N/A"
        print(f"{name:25s} | Acc: {res['accuracy']:.3f} (Δ {delta:+.3f}) | α: {alpha_str}")

    # Find best level
    best_level = max([k for k in level_results.keys() if k != "Baseline (No Levels)"],
                     key=lambda k: level_results[k]['accuracy'])
    print(f"\n✅ Best Level Config: {best_level}")
    print(f"   Accuracy: {level_results[best_level]['accuracy']:.3f}")

    # Key insights
    print("\n" + "="*70)
    print(" "*25 + "KEY INSIGHTS")
    print("="*70)

    print("\n1️⃣  Component Contributions:")
    a_only = component_results.get("Alignment Only (A)", {}).get('accuracy', 0)
    s_only = component_results.get("Separation Only (S)", {}).get('accuracy', 0)
    c_only = component_results.get("Cohesion Only (C)", {}).get('accuracy', 0)
    full = component_results.get("A + S + C (Full)", {}).get('accuracy', 0)

    if a_only > s_only and a_only > c_only:
        print("   ✅ Alignment (A) is the strongest individual component")
    elif s_only > a_only and s_only > c_only:
        print("   ✅ Separation (S) is the strongest individual component")
    elif c_only > a_only and c_only > s_only:
        print("   ✅ Cohesion (C) is the strongest individual component")

    if full > max(a_only, s_only, c_only):
        print("   ✅ Components are complementary (full > individual)")
    else:
        print("   ⚠️  Components may be redundant")

    print("\n2️⃣  Level Contributions:")
    l1 = level_results.get("L1: Graph Only", {}).get('accuracy', 0)
    l2 = level_results.get("L2: Kernel Only", {}).get('accuracy', 0)
    l3 = level_results.get("L3: Value Only", {}).get('accuracy', 0)
    full_level = level_results.get("L1 + L2 + L3 (Full)", {}).get('accuracy', 0)

    if l2 > l1 and l2 > l3:
        print("   ✅ Level 2 (Vector Kernel) is the strongest level")
    elif l1 > l2 and l1 > l3:
        print("   ✅ Level 1 (Graph) is the strongest level")
    elif l3 > l1 and l3 > l2:
        print("   ✅ Level 3 (Value) is the strongest level")

    if full_level > max(l1, l2, l3):
        print("   ✅ Levels are complementary (full > individual)")

    print("\n3️⃣  α Behavior:")
    alphas = [res['alpha'] for res in component_results.values() if res['alpha'] is not None]
    if alphas:
        alpha_mean = np.mean(alphas)
        alpha_std = np.std(alphas)
        print(f"   Mean α: {alpha_mean:.4f} ± {alpha_std:.4f}")
        if alpha_mean > 0.55:
            print("   → Models generally prefer learned attention")
        elif alpha_mean < 0.45:
            print("   → Models generally prefer bias attention")
        else:
            print("   → Balanced between learned and bias")

    # Save results
    print("\n" + "="*70)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results = {
        'component_ablation': component_results,
        'level_ablation': level_results,
    }

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"💾 Results saved to {output_dir / 'ablation_results.json'}")

    # Paper-ready summary
    print("\n" + "="*70)
    print(" "*20 + "PAPER-READY SUMMARY")
    print("="*70)

    print("\n📝 For Table 1 (Component Ablation):")
    print(f"   | Component | Accuracy | Δ vs Baseline | α     |")
    print(f"   |-----------|----------|---------------|-------|")
    for name in ["Baseline (No ASC)", "Alignment Only (A)", "Separation Only (S)",
                 "Cohesion Only (C)", "A + S", "A + C", "S + C", "A + S + C (Full)"]:
        if name in component_results:
            res = component_results[name]
            delta = res['accuracy'] - baseline_acc
            alpha_str = f"{res['alpha']:.4f}" if res['alpha'] else "  -  "
            print(f"   | {name:13s} | {res['accuracy']:.3f}    | {delta:+.3f}         | {alpha_str} |")

    print("\n📝 For Table 2 (Level Ablation):")
    print(f"   | Level Config    | Accuracy | Δ vs Baseline | α     |")
    print(f"   |-----------------|----------|---------------|-------|")
    for name in ["Baseline (No Levels)", "L1: Graph Only", "L2: Kernel Only",
                 "L3: Value Only", "L1 + L2", "L2 + L3", "L1 + L2 + L3 (Full)"]:
        if name in level_results:
            res = level_results[name]
            delta = res['accuracy'] - baseline_acc
            alpha_str = f"{res['alpha']:.4f}" if res['alpha'] else "  -  "
            print(f"   | {name:15s} | {res['accuracy']:.3f}    | {delta:+.3f}         | {alpha_str} |")

    print("\n" + "="*70)
    print("✅ Ablation study complete!")
    print("="*70)


if __name__ == "__main__":
    run_ablation_study()
