#!/bin/bash
#
# Complete Statistical Validation for ASCender v2.0
#
# Runs 42 experiments total:
# - 40 main experiments: h32/h48/h64/h80 × baseline/ascender × 5 seeds
# - 2 gradient analysis: h48/h64 (seed=42)
#
# Estimated time: 3-4 hours with 40+ parallel GPU instances
# Estimated cost: ~$80-100 (40 × g4dn.xlarge @ $0.526/hr × 3 hrs)
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

AWS_REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"  # $0.526/hr, 1x T4 GPU (16GB)
S3_BUCKET="s3://your-bucket-name/ascender-complete"

# Seeds for statistical validation
SEEDS=(42 123 456 789 2024)

# Hidden dimensions to test
HIDDEN_DIMS=(32 48 64 80)

echo "============================================================================"
echo "ASCender v2.0 - Complete Statistical Validation"
echo "============================================================================"
echo ""
echo "This will run:"
echo "  - 40 main experiments (4 sizes × 2 configs × 5 seeds)"
echo "  - 2 gradient analysis experiments (h48, h64)"
echo "  - Total: 42 parallel GPU instances"
echo ""
echo "Estimated cost: \$80-100 for 3 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# Step 1: Upload data to S3
# ============================================================================

echo ""
echo "Step 1: Uploading preprocessed data to S3..."
echo "============================================================================"

aws s3 sync data/ModelNet40/preprocessed/ ${S3_BUCKET}/data/ModelNet40/preprocessed/ \
    --region ${AWS_REGION}

aws s3 sync src/ ${S3_BUCKET}/src/ --region ${AWS_REGION}
aws s3 sync experiments/ ${S3_BUCKET}/experiments/ --region ${AWS_REGION}

echo "✅ Data uploaded"

# ============================================================================
# Step 2: Generate experiment list
# ============================================================================

echo ""
echo "Step 2: Generating experiment configurations..."
echo "============================================================================"

declare -a EXPERIMENTS=()

# Main experiments: 40 total
for h in "${HIDDEN_DIMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Baseline
        EXPERIMENTS+=("h${h}_baseline_seed${seed}:${h}:0:${seed}:main")
        # ASCender
        EXPERIMENTS+=("h${h}_ascender_seed${seed}:${h}:1:${seed}:main")
    done
done

# Gradient analysis: 2 total (h48, h64 with seed=42)
EXPERIMENTS+=("gradient_h48_seed42:48:0:42:gradient")
EXPERIMENTS+=("gradient_h64_seed42:64:0:42:gradient")

echo "Total experiments: ${#EXPERIMENTS[@]}"

# ============================================================================
# Step 3: Launch instances
# ============================================================================

echo ""
echo "Step 3: Launching EC2 instances..."
echo "============================================================================"

# NOTE: Instead of launching 42 instances at once, we'll use Terraform
# with Auto Scaling Groups for better management

echo "⚠️  TERRAFORM RECOMMENDED"
echo ""
echo "For 42 parallel experiments, it's recommended to use Terraform:"
echo ""
echo "  cd terraform/"
echo "  terraform apply -var 'desired_capacity=42'"
echo ""
echo "This will:"
echo "  1. Launch 42 spot instances (capacity-optimized)"
echo "  2. Auto-sync code/data from S3"
echo "  3. Run experiments in parallel"
echo "  4. Upload results to S3"
echo "  5. Terminate instances when complete"
echo ""
echo "Alternatively, run experiments manually with:"
echo ""

# Print example commands
echo "# Example: Run h32 baseline seed 42"
echo "python experiments/run_single_seed.py \\"
echo "  --hidden-dim 32 --seed 42 \\"
echo "  --preprocessed-dir data/ModelNet40/preprocessed \\"
echo "  --output-dir results/statistical_validation"
echo ""
echo "# Example: Run h48 gradient analysis"
echo "python experiments/run_gradient_analysis.py \\"
echo "  --hidden-dim 48 --seed 42 \\"
echo "  --preprocessed-dir data/ModelNet40/preprocessed \\"
echo "  --output-dir results/gradient_analysis"
echo ""

# ============================================================================
# Step 4: Save experiment manifest
# ============================================================================

MANIFEST_FILE="experiments_manifest.txt"
echo "# ASCender v2.0 Complete Validation - Experiment Manifest" > ${MANIFEST_FILE}
echo "# Generated: $(date)" >> ${MANIFEST_FILE}
echo "# Total experiments: ${#EXPERIMENTS[@]}" >> ${MANIFEST_FILE}
echo "" >> ${MANIFEST_FILE}

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name h use_asc seed type <<< "$exp"
    echo "$name|h=$h|ascender=$use_asc|seed=$seed|type=$type" >> ${MANIFEST_FILE}
done

echo "✅ Experiment manifest saved to: ${MANIFEST_FILE}"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "NEXT STEPS"
echo "============================================================================"
echo ""
echo "1. Review experiment manifest: cat ${MANIFEST_FILE}"
echo ""
echo "2. Launch experiments using Terraform:"
echo "   cd terraform/"
echo "   terraform plan"
echo "   terraform apply"
echo ""
echo "3. Monitor progress:"
echo "   aws s3 ls ${S3_BUCKET}/results/statistical_validation/"
echo "   aws s3 ls ${S3_BUCKET}/results/gradient_analysis/"
echo ""
echo "4. Download results when complete:"
echo "   bash aws_download_results.sh"
echo ""
echo "5. Generate statistical summary:"
echo "   python experiments/compute_statistics.py"
echo ""
echo "============================================================================"
echo "Preparation complete!"
echo "============================================================================"
echo ""
