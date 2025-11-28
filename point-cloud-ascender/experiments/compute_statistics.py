"""
Compute statistics from multiple seed results

Loads results from results/statistical_validation/ and computes:
- Mean ± standard deviation
- 95% confidence intervals
- Paired t-tests
- Statistical significance

Usage:
    python experiments/compute_statistics.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats


def load_results(hidden_dim, use_ascender):
    """Load all results for a given configuration"""
    results_dir = Path('results/statistical_validation')
    config_name = 'ascender' if use_ascender else 'baseline'

    results = []
    for seed in [42, 123, 456]:
        filename = f"h{hidden_dim}_{config_name}_seed{seed}.json"
        filepath = results_dir / filename

        if not filepath.exists():
            print(f"⚠️  Warning: Missing {filename}")
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results


def compute_statistics(results_list):
    """Compute mean, std, and confidence interval from multiple runs"""
    if not results_list:
        return None

    best_accs = [r['best_test_acc'] for r in results_list]

    mean = np.mean(best_accs)
    std = np.std(best_accs, ddof=1)  # Sample std (n-1)

    # 95% confidence interval
    n = len(best_accs)
    se = std / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se)

    return {
        'mean': mean,
        'std': std,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'individual_results': best_accs,
        'n_runs': n
    }


def perform_paired_ttest(baseline_results, ascender_results):
    """Perform paired t-test between baseline and ASCender"""
    baseline_accs = [r['best_test_acc'] for r in baseline_results]
    ascender_accs = [r['best_test_acc'] for r in ascender_results]

    # Ensure same number of runs
    min_len = min(len(baseline_accs), len(ascender_accs))
    baseline_accs = baseline_accs[:min_len]
    ascender_accs = ascender_accs[:min_len]

    if min_len < 2:
        return None

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(ascender_accs, baseline_accs)

    # Effect size (Cohen's d)
    diff = np.array(ascender_accs) - np.array(baseline_accs)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'significant_at_0.001': p_value < 0.001
    }


def format_pvalue(p):
    """Format p-value with appropriate precision"""
    if p < 0.001:
        return "< 0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f} (n.s.)"


def main():
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - MODELNET40 VALIDATION")
    print("="*80 + "\n")

    all_results = {}

    for hidden_dim in [32, 80]:
        print(f"Processing h{hidden_dim}...")

        baseline_results = load_results(hidden_dim, False)
        ascender_results = load_results(hidden_dim, True)

        if not baseline_results or not ascender_results:
            print(f"  ⚠️  Insufficient results for h{hidden_dim}, skipping...\n")
            continue

        baseline_stats = compute_statistics(baseline_results)
        ascender_stats = compute_statistics(ascender_results)
        ttest_results = perform_paired_ttest(baseline_results, ascender_results)

        if not ttest_results:
            print(f"  ⚠️  Cannot perform t-test for h{hidden_dim}, skipping...\n")
            continue

        all_results[f'h{hidden_dim}'] = {
            'baseline': baseline_stats,
            'ascender': ascender_stats,
            'paired_ttest': ttest_results,
            'delta_mean': (ascender_stats['mean'] - baseline_stats['mean']) * 100,
            'baseline_mean_pct': baseline_stats['mean'] * 100,
            'ascender_mean_pct': ascender_stats['mean'] * 100
        }

        print(f"  ✅ h{hidden_dim} analysis complete\n")

    # Save summary statistics
    output_dir = Path('results/statistical_validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Summary saved to: {summary_path}\n")

    # Print summary table
    print("="*80)
    print("STATISTICAL VALIDATION RESULTS")
    print("="*80 + "\n")

    table_lines = []
    table_lines.append("="*80)
    table_lines.append(f"{'Model':<10} {'Config':<10} {'Mean Acc':<15} {'Std':<10} {'Δ Acc':<10} {'p-value':<15}")
    table_lines.append("="*80)

    for model_size, stats in all_results.items():
        baseline_mean = stats['baseline_mean_pct']
        baseline_std = stats['baseline']['std'] * 100
        ascender_mean = stats['ascender_mean_pct']
        ascender_std = stats['ascender']['std'] * 100
        delta = stats['delta_mean']
        p_value = stats['paired_ttest']['p_value']
        p_formatted = format_pvalue(p_value)

        # Baseline row
        table_lines.append(f"{model_size:<10} {'Baseline':<10} {baseline_mean:>6.2f}% ± {baseline_std:>4.2f}%  {'':<10} {'':<10} {'':<15}")

        # ASCender row
        table_lines.append(f"{model_size:<10} {'ASCender':<10} {ascender_mean:>6.2f}% ± {ascender_std:>4.2f}%  {'':<10} {delta:>+6.2f}%    {p_formatted:<15}")

        table_lines.append("-"*80)

    table_lines.append("="*80)
    table_lines.append("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, (n.s.) not significant")
    table_lines.append("="*80)

    table_text = "\n".join(table_lines)
    print(table_text)

    # Save table to file
    table_path = output_dir / 'summary_table.txt'
    with open(table_path, 'w') as f:
        f.write(table_text)

    print(f"\n✅ Table saved to: {table_path}\n")

    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80 + "\n")

    for model_size, stats in all_results.items():
        print(f"{model_size.upper()}:")
        print("-"*80)

        baseline_mean = stats['baseline_mean_pct']
        baseline_std = stats['baseline']['std'] * 100
        baseline_ci = (stats['baseline']['ci_95_lower'] * 100, stats['baseline']['ci_95_upper'] * 100)

        ascender_mean = stats['ascender_mean_pct']
        ascender_std = stats['ascender']['std'] * 100
        ascender_ci = (stats['ascender']['ci_95_lower'] * 100, stats['ascender']['ci_95_upper'] * 100)

        delta = stats['delta_mean']
        t_stat = stats['paired_ttest']['t_statistic']
        p_value = stats['paired_ttest']['p_value']
        cohens_d = stats['paired_ttest']['cohens_d']

        print(f"  Baseline:  {baseline_mean:.2f}% ± {baseline_std:.2f}%")
        print(f"             95% CI: [{baseline_ci[0]:.2f}%, {baseline_ci[1]:.2f}%]")
        print(f"             Individual: {[f'{x*100:.2f}%' for x in stats['baseline']['individual_results']]}")
        print()
        print(f"  ASCender:  {ascender_mean:.2f}% ± {ascender_std:.2f}%")
        print(f"             95% CI: [{ascender_ci[0]:.2f}%, {ascender_ci[1]:.2f}%]")
        print(f"             Individual: {[f'{x*100:.2f}%' for x in stats['ascender']['individual_results']]}")
        print()
        print(f"  Δ Accuracy: {delta:+.2f}%")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print()

        if stats['paired_ttest']['significant_at_0.05']:
            winner = "ASCender" if delta > 0 else "Baseline"
            sig_level = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            print(f"  ✅ Result: {winner} wins (statistically significant {sig_level})")

            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect_size = "negligible"
            elif abs_d < 0.5:
                effect_size = "small"
            elif abs_d < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"
            print(f"  Effect size: {effect_size} (|d| = {abs_d:.2f})")
        else:
            print(f"  Result: No significant difference")

        print()

    print("="*80)
    print("✅ Statistical analysis complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
