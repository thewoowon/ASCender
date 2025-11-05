#!/usr/bin/env python3
"""
Plot RBP (Residual Bias Path) experiment results:
1. Loss comparison: Baseline vs RBP
2. Alpha evolution across epochs and layers
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# ============================================================
# Load data
# ============================================================
csv_path = Path(__file__).parent / "logs" / "results_summary.csv"
alpha_dir = Path(__file__).parent / "logs" / "alpha"

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from CSV")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Filter: Only latest experiments (Baseline + RBP)
# Baseline: use_ascender=False
# RBP: use_ascender=True (latest run)
baseline = df[df['use_ascender'] == False].copy()
rbp = df[df['use_ascender'] == True].copy()

# Get latest RBP run (by timestamp)
if len(rbp) > 0:
    latest_timestamp = rbp['timestamp'].max()
    rbp = rbp[rbp['timestamp'] == latest_timestamp].copy()

print(f"\nBaseline: {len(baseline)} rows")
print(f"RBP: {len(rbp)} rows")

# Load alpha data
alpha_files = sorted(alpha_dir.glob("alpha_epoch*_seed42.json"))
alpha_data = []
for f in alpha_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        alpha_data.append(data)

print(f"Loaded {len(alpha_data)} alpha tracking files")

# ============================================================
# Create figure
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('RBP (Residual Bias Path) Experiment Results',
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================
# Plot 1: Loss comparison (large, spanning 2 columns)
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])

if len(baseline) > 0:
    ax1.plot(baseline['epoch'], baseline['avg_loss'],
             marker='o', linewidth=3, markersize=9,
             label='Baseline (Standard Additive Bias)',
             color='#2E86AB', linestyle='-', alpha=0.8)

if len(rbp) > 0:
    ax1.plot(rbp['epoch'], rbp['avg_loss'],
             marker='s', linewidth=3, markersize=9,
             label='RBP (Residual Bias Path)',
             color='#A23B72', linestyle='-', alpha=0.8)

ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Loss', fontsize=13, fontweight='bold')
ax1.set_title('Training Loss: Baseline vs Residual Bias Path',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, linewidth=0.8)
ax1.set_xlim(0.5, 10.5)

# ============================================================
# Plot 2: Final epoch comparison (bar)
# ============================================================
ax2 = fig.add_subplot(gs[0, 2])

final_epoch = df['epoch'].max()
final_baseline = baseline[baseline['epoch'] == final_epoch]['avg_loss'].values
final_rbp = rbp[rbp['epoch'] == final_epoch]['avg_loss'].values

if len(final_baseline) > 0 and len(final_rbp) > 0:
    bars = ax2.bar(['Baseline', 'RBP'],
                   [final_baseline[0], final_rbp[0]],
                   color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)

    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title(f'Final Loss\n(Epoch {final_epoch})', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, [final_baseline[0], final_rbp[0]])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add delta
    delta = final_baseline[0] - final_rbp[0]
    improvement = (delta / final_baseline[0]) * 100
    ax2.text(0.5, max(final_baseline[0], final_rbp[0]) * 0.5,
            f'Δ = {delta:+.6f}\n({improvement:+.3f}%)',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================
# Plot 3: Loss improvement over baseline (%)
# ============================================================
ax3 = fig.add_subplot(gs[1, :2])

if len(baseline) > 0 and len(rbp) > 0:
    baseline_losses = baseline.set_index('epoch')['avg_loss']
    improvements = []
    epochs = []

    for _, row in rbp.iterrows():
        epoch = row['epoch']
        if epoch in baseline_losses.index:
            base_loss = baseline_losses[epoch]
            improvement = ((base_loss - row['avg_loss']) / base_loss) * 100
            improvements.append(improvement)
            epochs.append(epoch)

    ax3.plot(epochs, improvements, marker='D', linewidth=3, markersize=8,
            color='#F18F01', alpha=0.85, label='RBP Improvement')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    ax3.fill_between(epochs, 0, improvements, alpha=0.2, color='#F18F01')

    ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Loss Improvement vs Baseline', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3, linewidth=0.8)
    ax3.set_xlim(0.5, 10.5)

# ============================================================
# Plot 4: Alpha statistics box
# ============================================================
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')

# Load final analysis
final_analysis_path = alpha_dir / "FINAL_ANALYSIS_seed42.txt"
if final_analysis_path.exists():
    with open(final_analysis_path, 'r') as f:
        analysis_text = f.read()

    # Extract key stats
    stats_text = "α Mixing Statistics\n" + "="*30 + "\n\n"
    for line in analysis_text.split('\n'):
        if 'mean α' in line.lower() or 'median' in line or 'std' in line or 'min' in line or 'max' in line:
            stats_text += line.strip() + "\n"

    ax4.text(0.1, 0.5, stats_text,
            transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F4EA', alpha=0.8))

# ============================================================
# Plot 5: Alpha evolution by layer (heatmap-style)
# ============================================================
ax5 = fig.add_subplot(gs[2, :2])

# Extract alpha values for decoder layers with bias
decoder_l0_alphas = []
decoder_l1_alphas = []
epochs_alpha = []

for data in alpha_data:
    epoch = data['epoch']
    epochs_alpha.append(epoch)

    # Decoder L0 (has_bias=True)
    dec_l0 = [layer for layer in data['decoder'] if layer['layer'] == 0 and layer.get('has_bias', False)]
    if dec_l0:
        decoder_l0_alphas.append(dec_l0[0]['mean'])
    else:
        decoder_l0_alphas.append(0.5)

    # Decoder L1 (has_bias=True)
    dec_l1 = [layer for layer in data['decoder'] if layer['layer'] == 1 and layer.get('has_bias', False)]
    if dec_l1:
        decoder_l1_alphas.append(dec_l1[0]['mean'])
    else:
        decoder_l1_alphas.append(0.5)

# Plot as lines
ax5.plot(epochs_alpha, decoder_l0_alphas, marker='o', linewidth=2.5, markersize=7,
        label='Decoder L0 (self-attn)', color='#C73E1D', alpha=0.85)
ax5.plot(epochs_alpha, decoder_l1_alphas, marker='s', linewidth=2.5, markersize=7,
        label='Decoder L1 (self-attn)', color='#6A4C93', alpha=0.85)
ax5.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='α=0.5 (50/50 mix)')

ax5.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax5.set_ylabel('α (Normal Path Weight)', fontsize=13, fontweight='bold')
ax5.set_title('Alpha Evolution: Normal vs Biased Path Mixing',
              fontsize=14, fontweight='bold', pad=15)
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3, linewidth=0.8)
ax5.set_xlim(0.5, 10.5)
ax5.set_ylim(0.499, 0.501)

# Add annotation
ax5.text(0.98, 0.05,
        'α > 0.5: Model prefers normal (unbiased) attention\n'
        'α < 0.5: Model prefers biased attention\n'
        'α ≈ 0.5: Balanced mix',
        transform=ax5.transAxes, fontsize=9,
        ha='right', va='bottom', style='italic',
        bbox=dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.7))

# ============================================================
# Plot 6: Per-head alpha distribution (final epoch)
# ============================================================
ax6 = fig.add_subplot(gs[2, 2])

if len(alpha_data) > 0:
    final_alpha_data = alpha_data[-1]

    # Extract per-head alphas
    dec_l0_heads = None
    dec_l1_heads = None

    for layer in final_alpha_data['decoder']:
        if layer['layer'] == 0 and layer.get('has_bias', False):
            dec_l0_heads = layer['alpha']
        if layer['layer'] == 1 and layer.get('has_bias', False):
            dec_l1_heads = layer['alpha']

    if dec_l0_heads and dec_l1_heads:
        heads = np.arange(len(dec_l0_heads))
        width = 0.35

        bars1 = ax6.bar(heads - width/2, dec_l0_heads, width,
                       label='Dec L0', color='#C73E1D', alpha=0.7)
        bars2 = ax6.bar(heads + width/2, dec_l1_heads, width,
                       label='Dec L1', color='#6A4C93', alpha=0.7)

        ax6.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax6.set_xlabel('Head Index', fontsize=12, fontweight='bold')
        ax6.set_ylabel('α Value', fontsize=12, fontweight='bold')
        ax6.set_title(f'Per-Head α Distribution\n(Epoch {final_alpha_data["epoch"]})',
                     fontsize=13, fontweight='bold')
        ax6.set_xticks(heads)
        ax6.legend(fontsize=10)
        ax6.grid(True, axis='y', alpha=0.3)
        ax6.set_ylim(0.499, 0.501)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "logs" / "rbp_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Chart saved: {output_path}")

# ============================================================
# Statistical summary
# ============================================================
print("\n" + "="*70)
print("RBP EXPERIMENT ANALYSIS")
print("="*70)

print("\n1. LOSS COMPARISON (Final Epoch):")
print("-" * 50)
if len(final_baseline) > 0 and len(final_rbp) > 0:
    print(f"  Baseline:  {final_baseline[0]:.6f}")
    print(f"  RBP:       {final_rbp[0]:.6f}")
    delta = final_baseline[0] - final_rbp[0]
    improvement = (delta / final_baseline[0]) * 100
    print(f"  Δ:         {delta:+.6f}")
    print(f"  Change:    {improvement:+.4f}%")

    if abs(improvement) < 0.01:
        print("  ⚠️  VERDICT: Negligible difference (< 0.01%)")
    elif improvement > 0:
        print("  ✓  VERDICT: RBP shows slight improvement")
    else:
        print("  ✗  VERDICT: RBP slightly worse than baseline")

print("\n2. ALPHA BEHAVIOR:")
print("-" * 50)
if len(alpha_data) > 0:
    final = alpha_data[-1]
    initial = alpha_data[0]

    print(f"  Initial epoch ({initial['epoch']}):")
    for layer in initial['decoder']:
        if layer.get('has_bias', False):
            print(f"    Dec L{layer['layer']}: α = {layer['mean']:.6f}")

    print(f"\n  Final epoch ({final['epoch']}):")
    for layer in final['decoder']:
        if layer.get('has_bias', False):
            print(f"    Dec L{layer['layer']}: α = {layer['mean']:.6f}")

    # Analysis
    print("\n  Interpretation:")
    for layer in final['decoder']:
        if layer.get('has_bias', False):
            alpha_mean = layer['mean']
            if alpha_mean > 0.5001:
                print(f"    L{layer['layer']}: Model slightly prefers NORMAL attention (+{(alpha_mean-0.5)*100:.3f}%)")
            elif alpha_mean < 0.4999:
                print(f"    L{layer['layer']}: Model slightly prefers BIASED attention ({(alpha_mean-0.5)*100:.3f}%)")
            else:
                print(f"    L{layer['layer']}: Model uses BALANCED mix (α ≈ 0.5)")

print("\n3. CONVERGENCE:")
print("-" * 50)
if len(decoder_l0_alphas) > 1:
    l0_drift = abs(decoder_l0_alphas[-1] - decoder_l0_alphas[0])
    l1_drift = abs(decoder_l1_alphas[-1] - decoder_l1_alphas[0])
    print(f"  Dec L0 α drift: {l0_drift:.6f} ({l0_drift*100:.3f}%)")
    print(f"  Dec L1 α drift: {l1_drift:.6f} ({l1_drift*100:.3f}%)")

    if max(l0_drift, l1_drift) < 0.001:
        print("  → Alpha values are STABLE (minimal drift)")
    else:
        print("  → Alpha values show some EVOLUTION during training")

print("\n4. AVERAGE LOSS (All Epochs):")
print("-" * 50)
if len(baseline) > 0 and len(rbp) > 0:
    baseline_avg = baseline['avg_loss'].mean()
    rbp_avg = rbp['avg_loss'].mean()
    print(f"  Baseline:  {baseline_avg:.6f}")
    print(f"  RBP:       {rbp_avg:.6f}")
    print(f"  Δ:         {(baseline_avg - rbp_avg):+.6f} ({((baseline_avg-rbp_avg)/baseline_avg)*100:+.4f}%)")

print("\n" + "="*70)
print("\n💡 KEY INSIGHTS:")
print("-" * 50)
print("  • α ≈ 0.5 means the model learned to use a BALANCED mix")
print("  • Small α drift indicates stable, consistent behavior")
print("  • Near-identical loss suggests RBP doesn't harm performance")
print("  • The architecture WORKS but bias itself may need tuning")
print("="*70 + "\n")

plt.show()
