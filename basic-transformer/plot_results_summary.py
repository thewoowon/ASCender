#!/usr/bin/env python3
"""
Plot results_summary.csv: avg_loss by use_ascender and bias_combo
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Read data
csv_path = Path(__file__).parent / "logs" / "results_summary.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nUnique combinations:")
print(df.groupby(['use_ascender', 'bias_combo'])['epoch'].count())

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ASCender Experiment Results: Loss Analysis', fontsize=16, fontweight='bold')

# ============================================================
# Plot 1: Loss curves by bias_combo (use_ascender=True vs False)
# ============================================================
ax1 = axes[0, 0]

# Baseline (use_ascender=False)
baseline = df[df['use_ascender'] == False]
if len(baseline) > 0:
    ax1.plot(baseline['epoch'], baseline['avg_loss'],
             marker='o', linewidth=3, markersize=8,
             label='Baseline (no ASCender)', color='black', linestyle='--', alpha=0.7)

# ASCender experiments (use_ascender=True)
ascender = df[df['use_ascender'] == True]
bias_combos = ascender['bias_combo'].unique()

# Color palette
colors = plt.cm.tab10(np.linspace(0, 1, len(bias_combos)))
for i, combo in enumerate(sorted(bias_combos)):
    subset = ascender[ascender['bias_combo'] == combo]
    ax1.plot(subset['epoch'], subset['avg_loss'],
             marker='o', linewidth=2.5, markersize=7,
             label=f'ASCender ({combo})', color=colors[i], alpha=0.85)

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Average Loss', fontsize=12)
ax1.set_title('Training Loss: Baseline vs ASCender Variants', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# ============================================================
# Plot 2: Final epoch loss comparison (bar chart)
# ============================================================
ax2 = axes[0, 1]

# Get final epoch (epoch 10) for each configuration
final_epoch = df['epoch'].max()
final_results = df[df['epoch'] == final_epoch].copy()
final_results['config'] = final_results.apply(
    lambda x: 'Baseline' if not x['use_ascender'] else f"{x['bias_combo']}", axis=1
)
final_results = final_results.sort_values('avg_loss')

# Create bar chart
bar_colors = plt.cm.coolwarm(np.linspace(0, 1, len(final_results)))
bars = ax2.barh(range(len(final_results)), final_results['avg_loss'],
                color=bar_colors)
ax2.set_yticks(range(len(final_results)))
ax2.set_yticklabels(final_results['config'])
ax2.set_xlabel('Average Loss (Final Epoch)', fontsize=12)
ax2.set_title(f'Final Loss Comparison (Epoch {final_epoch})', fontsize=13, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(final_results.iterrows()):
    ax2.text(row['avg_loss'] + 0.01, i, f"{row['avg_loss']:.4f}",
             va='center', fontsize=9, fontweight='bold')

# ============================================================
# Plot 3: Loss improvement over baseline (%)
# ============================================================
ax3 = axes[1, 0]

# Calculate improvement for each epoch
baseline_losses = baseline.set_index('epoch')['avg_loss'] if len(baseline) > 0 else None
if baseline_losses is not None:
    for i, combo in enumerate(sorted(bias_combos)):
        subset = ascender[ascender['bias_combo'] == combo]
        improvements = []
        epochs = []
        for _, row in subset.iterrows():
            epoch = row['epoch']
            if epoch in baseline_losses.index:
                base_loss = baseline_losses[epoch]
                improvement = ((base_loss - row['avg_loss']) / base_loss) * 100
                improvements.append(improvement)
                epochs.append(epoch)

        ax3.plot(epochs, improvements, marker='o', linewidth=2.5, markersize=7,
                label=f'{combo}', color=colors[i], alpha=0.85)

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax3.set_title('Loss Improvement vs Baseline', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No baseline data available',
             ha='center', va='center', transform=ax3.transAxes, fontsize=14)

# ============================================================
# Plot 4: Learning speed (loss delta per epoch)
# ============================================================
ax4 = axes[1, 1]

# Calculate loss reduction rate
for i, combo in enumerate(sorted(bias_combos)):
    subset = ascender[ascender['bias_combo'] == combo].sort_values('epoch')
    if len(subset) > 1:
        loss_deltas = -subset['avg_loss'].diff().dropna()  # Negative of diff = reduction
        epochs = subset['epoch'].iloc[1:].values
        ax4.plot(epochs, loss_deltas, marker='o', linewidth=2.5, markersize=7,
                label=f'{combo}', color=colors[i], alpha=0.85)

# Baseline
if len(baseline) > 1:
    baseline_sorted = baseline.sort_values('epoch')
    baseline_deltas = -baseline_sorted['avg_loss'].diff().dropna()
    baseline_epochs = baseline_sorted['epoch'].iloc[1:].values
    ax4.plot(baseline_epochs, baseline_deltas, marker='o', linewidth=3, markersize=8,
            label='Baseline', color='black', linestyle='--', alpha=0.7)

ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Loss Reduction per Epoch', fontsize=12)
ax4.set_title('Learning Speed (Loss Δ per Epoch)', fontsize=13, fontweight='bold')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent / "logs" / "results_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Chart saved: {output_path}")

# ============================================================
# Additional: Statistical summary
# ============================================================
print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)

print("\nFinal Epoch Performance (Epoch 10):")
print("-" * 50)
for _, row in final_results.iterrows():
    config = row['config']
    loss = row['avg_loss']
    if config == 'Baseline':
        print(f"  {config:20s}: {loss:.6f}")
    else:
        base_loss = final_results[final_results['config'] == 'Baseline']['avg_loss'].values
        if len(base_loss) > 0:
            improvement = ((base_loss[0] - loss) / base_loss[0]) * 100
            delta = base_loss[0] - loss
            print(f"  ASCender ({config:12s}): {loss:.6f}  (Δ={delta:+.6f}, {improvement:+.3f}%)")
        else:
            print(f"  ASCender ({config:12s}): {loss:.6f}")

print("\nBest Performer:")
print("-" * 50)
best = final_results.iloc[0]
best_config = 'Baseline' if not best['use_ascender'] else f"ASCender ({best['bias_combo']})"
print(f"  Configuration: {best_config}")
print(f"  Final Loss: {best['avg_loss']:.6f}")

# Average loss across all epochs
print("\nAverage Loss (All Epochs):")
print("-" * 50)
avg_by_config = df.groupby(['use_ascender', 'bias_combo'])['avg_loss'].mean().sort_values()
for (use_asc, combo), avg_loss in avg_by_config.items():
    config_name = 'Baseline' if not use_asc else f'ASCender ({combo})'
    print(f"  {config_name:30s}: {avg_loss:.6f}")

print("\n" + "="*70)

plt.show()
