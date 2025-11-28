#!/bin/bash
#
# Download results from AWS S3 and compute statistics
#
# Usage:
#   bash aws_download_results.sh

set -e

# ============================================================================
# CONFIGURATION - MUST MATCH aws_parallel_launcher.sh
# ============================================================================

AWS_REGION="us-east-1"
S3_BUCKET="s3://your-bucket-name/ascender-experiments"

# ============================================================================
# Download results from S3
# ============================================================================

echo "============================================================================"
echo "Downloading results from AWS S3"
echo "============================================================================"
echo ""

mkdir -p results/statistical_validation

echo "Downloading from: ${S3_BUCKET}/results/statistical_validation/"
aws s3 sync ${S3_BUCKET}/results/statistical_validation/ results/statistical_validation/ \
    --region ${AWS_REGION}

echo ""
echo "✅ Results downloaded!"
echo ""

# ============================================================================
# Check which results are available
# ============================================================================

echo "============================================================================"
echo "Available results:"
echo "============================================================================"
ls -lh results/statistical_validation/*.json

echo ""

# Count results
NUM_RESULTS=$(ls results/statistical_validation/*.json 2>/dev/null | wc -l)
echo "Total results: ${NUM_RESULTS}/12"

if [ ${NUM_RESULTS} -lt 12 ]; then
    echo "⚠️  Warning: Not all experiments have completed yet."
    echo "   Expected: 12 results (3 seeds × 2 models × 2 configs)"
    echo "   Found: ${NUM_RESULTS} results"
    echo ""
    echo "Missing experiments:"

    # Check each expected result
    for hidden_dim in 32 80; do
        for config in baseline ascender; do
            for seed in 42 123 456; do
                filename="h${hidden_dim}_${config}_seed${seed}.json"
                if [ ! -f "results/statistical_validation/${filename}" ]; then
                    echo "  - ${filename}"
                fi
            done
        done
    done
    echo ""
fi

# ============================================================================
# Compute statistics if all results are available
# ============================================================================

if [ ${NUM_RESULTS} -eq 12 ]; then
    echo "============================================================================"
    echo "All results available! Computing statistics..."
    echo "============================================================================"
    echo ""

    python experiments/compute_statistics.py

    echo ""
    echo "✅ Statistical analysis complete!"
    echo ""
    echo "Results saved to:"
    echo "  - results/statistical_validation/summary.json"
    echo "  - results/statistical_validation/summary_table.txt"
else
    echo "Waiting for remaining experiments to complete..."
    echo "Re-run this script when more results are available."
fi

echo ""
echo "============================================================================"
echo "Done!"
echo "============================================================================"
