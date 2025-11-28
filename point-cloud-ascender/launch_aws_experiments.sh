#!/bin/bash
#
# Launch Complete ASCender Validation on AWS
#
# This script:
# 1. Uploads preprocessed data and code to S3
# 2. Launches 42 GPU spot instances via Terraform
# 3. Monitors progress
# 4. Downloads results when complete
#

set -e

BUCKET="ascender-research-20251005"
REGION="us-east-1"

echo "============================================================================"
echo "ASCender v2.0 - AWS Parallel Experiments Launcher"
echo "============================================================================"
echo ""
echo "This will:"
echo "  - Upload ~3GB preprocessed ModelNet40 data to S3"
echo "  - Launch 42 GPU spot instances (g4dn.xlarge)"
echo "  - Run experiments in parallel (estimated 2-3 hours)"
echo "  - Auto-shutdown instances when complete"
echo ""
echo "Estimated cost: \$20-30 (spot instances ~\$0.16/hr each)"
echo ""

# Auto-confirm if AUTO_CONFIRM=1
if [ "$AUTO_CONFIRM" != "1" ]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "Auto-confirming (AUTO_CONFIRM=1)..."
fi

# ============================================================================
# Step 1: Upload Data and Code to S3
# ============================================================================

echo ""
echo "============================================================================"
echo "Step 1: Uploading data and code to S3"
echo "============================================================================"
echo ""

# Check if preprocessed data exists
if [ ! -d "data/ModelNet40/preprocessed/train" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run preprocessing first:"
    echo "  python experiments/preprocess_modelnet40.py"
    exit 1
fi

# Upload preprocessed data
echo "Uploading preprocessed ModelNet40 data..."
aws s3 sync data/ModelNet40/preprocessed/ \
    s3://$BUCKET/data/ModelNet40/preprocessed/ \
    --region $REGION \
    --storage-class STANDARD_IA

echo "✅ Data uploaded"

# Upload source code
echo "Uploading source code..."
aws s3 sync src/ s3://$BUCKET/src/ --region $REGION --delete
aws s3 sync experiments/ s3://$BUCKET/experiments/ --region $REGION --delete

echo "✅ Code uploaded"

# ============================================================================
# Step 2: Launch Experiments with Terraform
# ============================================================================

echo ""
echo "============================================================================"
echo "Step 2: Launching GPU instances with Terraform"
echo "============================================================================"
echo ""

cd terraform/

# Initialize Terraform (if needed)
if [ ! -d ".terraform" ]; then
    echo "Initializing Terraform..."
    terraform init
fi

# Apply configuration
echo ""
echo "Launching 42 spot instances..."
echo ""

terraform apply \
    -var-file=terraform.tfvars \
    -target=aws_spot_instance_request.experiment \
    -auto-approve \
    main_parallel_v2.tf

echo ""
echo "✅ Instances launched!"
echo ""

# Get spot request IDs
SPOT_REQUESTS=$(terraform output -json spot_request_ids | jq -r '.[]')

echo "Spot request IDs:"
echo "$SPOT_REQUESTS"

cd ..

# ============================================================================
# Step 3: Monitor Progress
# ============================================================================

echo ""
echo "============================================================================"
echo "Step 3: Monitoring progress"
echo "============================================================================"
echo ""

echo "Experiments are running in the background."
echo ""
echo "Monitor progress with:"
echo ""
echo "  # Check how many experiments completed"
echo "  watch -n 10 'aws s3 ls s3://$BUCKET/results/completion_markers/ | wc -l'"
echo ""
echo "  # View experiment status"
echo "  aws s3 ls s3://$BUCKET/results/completion_markers/"
echo ""
echo "  # Check spot instance status"
echo "  aws ec2 describe-spot-instance-requests --region $REGION \\"
echo "    --filters Name=state,Values=active,open"
echo ""
echo "  # View instance console logs (for debugging)"
echo "  aws ec2 get-console-output --region $REGION --instance-id <INSTANCE_ID>"
echo ""

# ============================================================================
# Step 4: Wait for Completion (Optional)
# ============================================================================

echo ""

# Auto-skip waiting if AUTO_CONFIRM=1
if [ "$AUTO_CONFIRM" != "1" ]; then
    read -p "Wait for all experiments to complete? (y/n) " -n 1 -r
    echo
    WAIT_FOR_COMPLETION=$REPLY
else
    echo "Skipping wait (AUTO_CONFIRM=1)..."
    WAIT_FOR_COMPLETION="n"
fi

if [[ $WAIT_FOR_COMPLETION =~ ^[Yy]$ ]]; then
    echo ""
    echo "Waiting for experiments to complete..."
    echo "This may take 2-3 hours."
    echo ""

    TOTAL_EXPERIMENTS=42
    COMPLETED=0

    while [ $COMPLETED -lt $TOTAL_EXPERIMENTS ]; do
        COMPLETED=$(aws s3 ls s3://$BUCKET/results/completion_markers/ | wc -l | tr -d ' ')
        echo "[$(date +%H:%M:%S)] Completed: $COMPLETED / $TOTAL_EXPERIMENTS"

        if [ $COMPLETED -lt $TOTAL_EXPERIMENTS ]; then
            sleep 60  # Check every minute
        fi
    done

    echo ""
    echo "✅ All experiments complete!"
fi

# ============================================================================
# Step 5: Download Results
# ============================================================================

echo ""
echo "============================================================================"
echo "Step 5: Downloading results"
echo "============================================================================"
echo ""

mkdir -p results/statistical_validation
mkdir -p results/gradient_analysis
mkdir -p results/completion_markers

echo "Downloading results from S3..."
aws s3 sync s3://$BUCKET/results/ results/ --region $REGION

echo "✅ Results downloaded to: ./results/"

# ============================================================================
# Step 6: Generate Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "Step 6: Generating statistical summary"
echo "============================================================================"
echo ""

if [ -f "experiments/compute_statistics.py" ]; then
    python experiments/compute_statistics.py
    echo "✅ Summary generated: results/statistical_validation/summary.json"
else
    echo "⚠️  compute_statistics.py not found, skipping summary"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "✅ AWS EXPERIMENTS COMPLETE!"
echo "============================================================================"
echo ""
echo "Results location:"
echo "  - Local: ./results/"
echo "  - S3: s3://$BUCKET/results/"
echo ""
echo "Next steps:"
echo "  1. Review results: cat results/statistical_validation/summary.json"
echo "  2. Update paper with new statistics"
echo "  3. Clean up resources: cd terraform && terraform destroy"
echo ""
echo "Total cost: Check AWS Cost Explorer"
echo "  (Estimated: \$20-30 for ~42 instances × 2-3 hours)"
echo ""
echo "============================================================================"
