#!/bin/bash
#
# AWS Parallel Launcher for Statistical Validation
#
# This script launches 12 parallel experiments across AWS GPU instances
# Estimated time: 2-3 hours (with 12 parallel g4dn.xlarge instances)
#
# Prerequisites:
# 1. AWS CLI configured with credentials
# 2. S3 bucket for data/results sync
# 3. SSH key pair for EC2 access
#
# Usage:
#   bash aws_parallel_launcher.sh

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# AWS Configuration
AWS_REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"  # $0.526/hr, 1x T4 GPU (16GB)
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS with CUDA
KEY_NAME="your-key-name"  # Your SSH key pair name
SECURITY_GROUP="sg-xxxxxxxx"  # Security group ID with SSH access
SUBNET_ID="subnet-xxxxxxxx"  # Your VPC subnet ID

# S3 bucket for data/results sync
S3_BUCKET="s3://your-bucket-name/ascender-experiments"

# Number of parallel instances (12 experiments = 12 instances for max speed)
NUM_INSTANCES=12

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Define all 12 experiments
declare -a EXPERIMENTS=(
    "h32_baseline_seed42:32:0:42"
    "h32_baseline_seed123:32:0:123"
    "h32_baseline_seed456:32:0:456"
    "h32_ascender_seed42:32:1:42"
    "h32_ascender_seed123:32:1:123"
    "h32_ascender_seed456:32:1:456"
    "h80_baseline_seed42:80:0:42"
    "h80_baseline_seed123:80:0:123"
    "h80_baseline_seed456:80:0:456"
    "h80_ascender_seed42:80:1:42"
    "h80_ascender_seed123:80:1:123"
    "h80_ascender_seed456:80:1:456"
)

# ============================================================================
# STEP 0: Check prerequisites
# ============================================================================

echo "============================================================================"
echo "AWS Parallel Launcher for Statistical Validation"
echo "============================================================================"
echo ""
echo "This will launch ${NUM_INSTANCES} g4dn.xlarge instances"
echo "Estimated cost: \$$(echo "${NUM_INSTANCES} * 0.526 * 3" | bc) for 3 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# STEP 1: Upload preprocessed data and code to S3
# ============================================================================

echo ""
echo "Step 1: Uploading preprocessed data and code to S3..."
echo "============================================================================"

# Upload preprocessed ModelNet40 data
aws s3 sync data/ModelNet40/preprocessed/ ${S3_BUCKET}/data/ModelNet40/preprocessed/ \
    --region ${AWS_REGION}

# Upload source code
aws s3 sync src/ ${S3_BUCKET}/src/ --region ${AWS_REGION}
aws s3 sync experiments/ ${S3_BUCKET}/experiments/ --region ${AWS_REGION}

echo "✅ Data and code uploaded to S3"

# ============================================================================
# STEP 2: Create bootstrap script
# ============================================================================

cat > /tmp/aws_bootstrap.sh << 'EOF'
#!/bin/bash
set -e

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create working directory
mkdir -p ~/ascender
cd ~/ascender

# Download data and code from S3
aws s3 sync S3_BUCKET_PLACEHOLDER/data/ data/
aws s3 sync S3_BUCKET_PLACEHOLDER/src/ src/
aws s3 sync S3_BUCKET_PLACEHOLDER/experiments/ experiments/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy tqdm

# Run experiment (parameters will be replaced)
python experiments/run_single_seed.py \
    --hidden-dim HIDDEN_DIM_PLACEHOLDER \
    --seed SEED_PLACEHOLDER \
    USE_ASCENDER_PLACEHOLDER \
    --epochs 50 \
    --batch-size 32 \
    --preprocessed-dir data/ModelNet40/preprocessed \
    --output-dir results/statistical_validation

# Upload results to S3
aws s3 cp results/statistical_validation/ S3_BUCKET_PLACEHOLDER/results/statistical_validation/ --recursive

echo "✅ Experiment complete! Results uploaded to S3."
EOF

# ============================================================================
# STEP 3: Launch EC2 instances
# ============================================================================

echo ""
echo "Step 2: Launching ${NUM_INSTANCES} EC2 instances..."
echo "============================================================================"

declare -a INSTANCE_IDS=()

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name hidden_dim use_ascender seed <<< "${EXPERIMENTS[$i]}"

    echo "Launching instance $((i+1))/${NUM_INSTANCES}: ${exp_name}..."

    # Prepare bootstrap script for this experiment
    BOOTSTRAP_SCRIPT="/tmp/aws_bootstrap_${exp_name}.sh"
    cp /tmp/aws_bootstrap.sh ${BOOTSTRAP_SCRIPT}

    # Replace placeholders
    sed -i.bak "s|S3_BUCKET_PLACEHOLDER|${S3_BUCKET}|g" ${BOOTSTRAP_SCRIPT}
    sed -i.bak "s|HIDDEN_DIM_PLACEHOLDER|${hidden_dim}|g" ${BOOTSTRAP_SCRIPT}
    sed -i.bak "s|SEED_PLACEHOLDER|${seed}|g" ${BOOTSTRAP_SCRIPT}

    if [ "${use_ascender}" == "1" ]; then
        sed -i.bak "s|USE_ASCENDER_PLACEHOLDER|--use-ascender|g" ${BOOTSTRAP_SCRIPT}
    else
        sed -i.bak "s|USE_ASCENDER_PLACEHOLDER||g" ${BOOTSTRAP_SCRIPT}
    fi

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --region ${AWS_REGION} \
        --image-id ${AMI_ID} \
        --instance-type ${INSTANCE_TYPE} \
        --key-name ${KEY_NAME} \
        --security-group-ids ${SECURITY_GROUP} \
        --subnet-id ${SUBNET_ID} \
        --user-data file://${BOOTSTRAP_SCRIPT} \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ascender-${exp_name}},{Key=Project,Value=ASCender-StatValidation}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    INSTANCE_IDS+=("${INSTANCE_ID}")
    echo "  ✅ Launched: ${INSTANCE_ID}"

    # Small delay to avoid rate limiting
    sleep 2
done

echo ""
echo "✅ All ${NUM_INSTANCES} instances launched!"
echo ""
echo "Instance IDs:"
for id in "${INSTANCE_IDS[@]}"; do
    echo "  - ${id}"
done

# ============================================================================
# STEP 4: Monitor progress
# ============================================================================

echo ""
echo "============================================================================"
echo "Monitoring instance status..."
echo "============================================================================"
echo ""
echo "Instances will:"
echo "  1. Download data from S3 (~5 min)"
echo "  2. Install dependencies (~5 min)"
echo "  3. Run training (50 epochs, ~2-3 hours)"
echo "  4. Upload results to S3"
echo "  5. Terminate automatically (optional)"
echo ""
echo "You can monitor progress in AWS Console or use:"
echo "  aws ec2 describe-instances --instance-ids ${INSTANCE_IDS[0]} ${INSTANCE_IDS[1]} ..."
echo ""
echo "Results will be uploaded to: ${S3_BUCKET}/results/statistical_validation/"
echo ""
echo "To download results when complete:"
echo "  bash aws_download_results.sh"
echo ""

# Save instance IDs for later
echo "${INSTANCE_IDS[@]}" > /tmp/ascender_instance_ids.txt

echo "Instance IDs saved to: /tmp/ascender_instance_ids.txt"
echo ""
echo "============================================================================"
echo "🚀 Parallel experiments launched!"
echo "============================================================================"
