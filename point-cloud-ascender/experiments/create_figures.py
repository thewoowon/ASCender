"""
Create figures for paper from experimental results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_alpha_evolution():
    """Figure 1: α evolution over training"""

    # Load RBP results
    with open('results/rbp_results.json', 'r') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each configuration
    configs = ['Weak Bias', 'Strong Bias', 'Very Strong Bias']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for config, color in zip(configs, colors):
        if config in results:
            alpha_hist = results[config]['alpha_history']
            epochs = np.arange(20, 101, 20)
            ax.plot(epochs, alpha_hist, marker='o', linewidth=2,
                   label=config, color=color, markersize=8)

    # Reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='α=0.5 (balanced)')

    # Labels
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('α (Mixing Parameter)', fontsize=14, fontweight='bold')
    ax.set_title('RBP α Evolution: Learned vs Bias Attention', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.45, 0.65])

    # Save
    plt.tight_layout()
    plt.savefig('results/figure1_alpha_evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure1_alpha_evolution.pdf', bbox_inches='tight')
    print("✅ Figure 1 saved: α evolution")
    plt.close()


def plot_accuracy_comparison():
    """Figure 2: Accuracy comparison (Baseline vs ASCender)"""

    with open('results/rbp_results.json', 'r') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    baseline_acc = 0.445  # From experiments
    configs = ['Weak Bias', 'Strong Bias', 'Very Strong Bias']
    accuracies = [results[c]['accuracy'] for c in configs]
    alphas = [results[c]['alpha'] for c in configs]

    # Bar plot
    x = np.arange(len(configs) + 1)
    bars = ax.bar(x, [baseline_acc] + accuracies,
                  color=['gray', '#2ecc71', '#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add α values on bars
    for i, (bar, alpha) in enumerate(zip(bars[1:], alphas), 1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'α={alpha:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Labels
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy: Baseline vs ASCender Variants', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline'] + configs, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.40, 0.55])

    # Add improvement annotations
    for i, (bar, acc) in enumerate(zip(bars[1:], accuracies), 1):
        improvement = acc - baseline_acc
        ax.text(bar.get_x() + bar.get_width()/2., 0.41,
               f'+{improvement*100:.1f}%',
               ha='center', va='bottom', fontsize=10,
               color='green' if improvement > 0 else 'red',
               fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figure2_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure2_accuracy_comparison.pdf', bbox_inches='tight')
    print("✅ Figure 2 saved: Accuracy comparison")
    plt.close()


def plot_alpha_vs_bias_strength():
    """Figure 3: α vs Bias Strength"""

    with open('results/rbp_results.json', 'r') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    configs = ['Weak Bias', 'Strong Bias', 'Very Strong Bias']
    bias_strengths = [1.2, 5.0, 12.0]  # w_a + w_s + w_c
    alphas = [results[c]['alpha'] for c in configs]
    accuracies = [results[c]['accuracy'] for c in configs]

    # Scatter plot with size representing accuracy
    sizes = [(acc - 0.40) * 3000 for acc in accuracies]  # Scale for visibility
    scatter = ax.scatter(bias_strengths, alphas, s=sizes,
                        c=accuracies, cmap='RdYlGn',
                        edgecolors='black', linewidth=2, alpha=0.7)

    # Add labels
    for bias, alpha, config in zip(bias_strengths, alphas, configs):
        ax.annotate(config, (bias, alpha),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    # Reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='α=0.5')

    # Labels
    ax.set_xlabel('Total Bias Strength (w_a + w_s + w_c)', fontsize=14, fontweight='bold')
    ax.set_ylabel('α (Mixing Parameter)', fontsize=14, fontweight='bold')
    ax.set_title('α Adaptation to Bias Strength', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accuracy', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figure3_alpha_vs_bias.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure3_alpha_vs_bias.pdf', bbox_inches='tight')
    print("✅ Figure 3 saved: α vs Bias Strength")
    plt.close()


def create_summary_figure():
    """Figure 4: Summary - All key results"""

    with open('results/rbp_results.json', 'r') as f:
        results = json.load(f)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: α evolution
    ax1 = fig.add_subplot(gs[0, 0])
    configs = ['Weak Bias', 'Strong Bias', 'Very Strong Bias']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for config, color in zip(configs, colors):
        if config in results:
            alpha_hist = results[config]['alpha_history']
            epochs = np.arange(20, 101, 20)
            ax1.plot(epochs, alpha_hist, marker='o', linewidth=2,
                    label=config, color=color, markersize=6)

    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('α', fontsize=12, fontweight='bold')
    ax1.set_title('(a) α Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_acc = 0.445
    accuracies = [results[c]['accuracy'] for c in configs]

    x = np.arange(len(configs) + 1)
    bars = ax2.bar(x, [baseline_acc] + accuracies,
                   color=['gray', '#2ecc71', '#3498db', '#e74c3c'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baseline', 'Weak', 'Strong', 'V.Strong'], fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: α final values
    ax3 = fig.add_subplot(gs[1, 0])
    alphas = [results[c]['alpha'] for c in configs]
    bars = ax3.barh(configs, alphas, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Final α', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Final α Values', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # Add values
    for i, (bar, alpha) in enumerate(zip(bars, alphas)):
        ax3.text(alpha + 0.01, i, f'{alpha:.4f}',
                va='center', fontsize=11, fontweight='bold')

    # Subplot 4: Improvement
    ax4 = fig.add_subplot(gs[1, 1])
    improvements = [(results[c]['accuracy'] - baseline_acc) * 100 for c in configs]
    bars = ax4.bar(configs, improvements, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Improvement vs Baseline', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add values
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{imp:+.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('ASCender v2.0 - Summary of Key Results',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('results/figure4_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure4_summary.pdf', bbox_inches='tight')
    print("✅ Figure 4 saved: Summary")
    plt.close()


def main():
    print("="*70)
    print(" "*20 + "Creating Figures for Paper")
    print("="*70)

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Check if data exists
    if not Path("results/rbp_results.json").exists():
        print("❌ rbp_results.json not found!")
        print("   Please run experiments first.")
        return

    print("\n📊 Generating figures...")

    try:
        plot_alpha_evolution()
        plot_accuracy_comparison()
        plot_alpha_vs_bias_strength()
        create_summary_figure()

        print("\n" + "="*70)
        print("✅ All figures created successfully!")
        print("="*70)

        print("\nFiles created:")
        print("  - results/figure1_alpha_evolution.png/pdf")
        print("  - results/figure2_accuracy_comparison.png/pdf")
        print("  - results/figure3_alpha_vs_bias.png/pdf")
        print("  - results/figure4_summary.png/pdf")

        print("\n📝 Ready to include in paper!")

    except Exception as e:
        print(f"\n❌ Error creating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
