#!/bin/bash
# Quick script to check preprocessing progress

echo "===================================================================="
echo "ModelNet40 Preprocessing Progress"
echo "===================================================================="
echo ""

# Count preprocessed files
train_count=$(ls data/ModelNet40/preprocessed/train/*.pt 2>/dev/null | wc -l | tr -d ' ')
test_count=$(ls data/ModelNet40/preprocessed/test/*.pt 2>/dev/null | wc -l | tr -d ' ')

echo "Preprocessed files:"
echo "  Train: $train_count / 9840"
echo "  Test:  $test_count / 2468"
echo ""

# Calculate percentage
if [ "$train_count" -gt 0 ]; then
    train_pct=$(echo "scale=1; $train_count * 100 / 9840" | bc)
    echo "  Train progress: ${train_pct}%"
fi

if [ "$test_count" -gt 0 ]; then
    test_pct=$(echo "scale=1; $test_count * 100 / 2468" | bc)
    echo "  Test progress:  ${test_pct}%"
fi

echo ""

# Check if preprocessing is complete
if [ "$train_count" -eq 9840 ] && [ "$test_count" -eq 2468 ]; then
    echo "✅ Preprocessing COMPLETE!"
    echo ""

    # Show sample structure
    echo "Sample structure:"
    python3 -c "
import torch
sample = torch.load('data/ModelNet40/preprocessed/train/sample_0000.pt')
print('  Keys:', list(sample.keys()))
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.shape}, {v.dtype}')
"
    echo ""

    # Calculate storage size
    total_size=$(du -sh data/ModelNet40/preprocessed/ | cut -f1)
    echo "Total storage: $total_size"
    echo ""
    echo "Ready for fast training! Run:"
    echo "  python experiments/fast_modelnet40_experiment.py --epochs 1"
else
    echo "⏳ Preprocessing in progress..."

    # Estimate time remaining (if train is being processed)
    if [ "$train_count" -gt 10 ]; then
        remaining=$((9840 - train_count))
        # Assume 4.6 samples/sec
        time_remaining=$(echo "scale=1; $remaining / 4.6 / 60" | bc)
        echo "  Estimated time remaining: ${time_remaining} minutes"
    fi
fi

echo ""
echo "===================================================================="
