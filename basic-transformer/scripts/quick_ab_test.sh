#!/bin/bash
# Quick A/B test runner
# Compares baseline vs aggressive ASCender configs

set -e

echo "=========================================="
echo "ASCender A/B Testing Suite"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/ab_tests

echo "Step 1/3: Running baseline (no bias)..."
python -m src.train --config configs/baseline.yaml 2>&1 | tee logs/ab_tests/baseline.log

echo ""
echo "Step 2/3: Running moderate aggressive ASCender..."
python -m src.train --config configs/ascender_moderate_aggressive.yaml 2>&1 | tee logs/ab_tests/moderate_aggressive.log

echo ""
echo "Step 3/3: Running very aggressive ASCender..."
python -m src.train --config configs/ascender_very_aggressive.yaml 2>&1 | tee logs/ab_tests/very_aggressive.log

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""

# Extract final losses from CSV
echo "Extracting losses from results_summary.csv..."
python3 << 'EOF'
import csv
from collections import defaultdict

results = defaultdict(list)

with open('logs/results_summary.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['name'] if 'name' in row else (
            'baseline' if row['use_ascender'] == 'False' else
            row.get('bias_combo', 'unknown')
        )
        results[name].append(float(row['avg_loss']))

# Get last 3 epochs for each config
for name, losses in results.items():
    if len(losses) >= 3:
        recent = losses[-3:]
        print(f"{name:25s}: {recent[-1]:.4f} (last 3 epochs: {recent})")
    elif losses:
        print(f"{name:25s}: {losses[-1]:.4f}")

print("\nInterpretation:")
print("  - If moderate_aggressive ≈ baseline: Bias has minimal effect (expected)")
print("  - If very_aggressive >> baseline: Bias is working but too strong")
print("  - If very_aggressive ≈ baseline: Bias still has no effect → check code")
EOF

echo ""
echo "Full logs saved to logs/ab_tests/"
echo "Run diagnostic script for detailed analysis:"
echo "  python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml"
echo ""
