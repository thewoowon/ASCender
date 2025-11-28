#!/bin/bash
#
# Single Experiment Runner for ASCender
# Template variables: BUCKET, EXP_NAME, HIDDEN_DIM, USE_ASCENDER, SEED, EXP_TYPE
#

set -e
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "============================================================================"
echo "ASCender GPU Experiment: ${EXP_NAME}"
echo "============================================================================"
echo "Timestamp: $(date)"
echo "Instance ID: $(ec2-metadata --instance-id | cut -d' ' -f2)"
echo "Instance Type: $(ec2-metadata --instance-type | cut -d' ' -f2)"
echo "Availability Zone: $(ec2-metadata --availability-zone | cut -d' ' -f2)"
echo "============================================================================"

# Wait for instance to be fully ready
sleep 30

# ============================================================================
# Verify GPU
# ============================================================================

echo ""
echo "Checking GPU..."
nvidia-smi || { echo "ERROR: No GPU!"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "✅ GPU detected"

# ============================================================================
# Setup Working Directory
# ============================================================================

cd /home/ubuntu
mkdir -p /home/ubuntu/ascender
cd /home/ubuntu/ascender

# ============================================================================
# Download Data and Code from S3
# ============================================================================

echo ""
echo "Downloading from S3..."
echo "Bucket: s3://${BUCKET}"

# Download preprocessed data
echo "  - ModelNet40 preprocessed data..."
aws s3 sync "s3://${BUCKET}/data/ModelNet40/preprocessed/" \
    data/ModelNet40/preprocessed/ \
    --region us-east-1 \
    --no-progress \
    --only-show-errors

# Download code
echo "  - Source code..."
aws s3 sync "s3://${BUCKET}/src/" src/ \
    --region us-east-1 --no-progress --only-show-errors

aws s3 sync "s3://${BUCKET}/experiments/" experiments/ \
    --region us-east-1 --no-progress --only-show-errors

echo "✅ S3 download complete"

# ============================================================================
# Setup Conda Environment
# ============================================================================

echo ""
echo "Setting up Python environment..."

# Find conda
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "ERROR: Conda not found!"
    exit 1
fi

# Create environment
conda create -y -n asc python=3.10 2>/dev/null || true
conda activate asc

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 12.1..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install --quiet numpy scipy tqdm matplotlib

echo "✅ Environment ready"

# Verify CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# ============================================================================
# Run Experiment
# ============================================================================

echo ""
echo "============================================================================"
echo "STARTING EXPERIMENT: ${EXP_NAME}"
echo "============================================================================"
echo "Type: ${EXP_TYPE}"
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "Seed: ${SEED}"
echo "ASCender: ${USE_ASCENDER}"
echo "============================================================================"
echo ""

cd /home/ubuntu/ascender

START_TIME=$(date +%s)

if [ "${EXP_TYPE}" = "main" ]; then
    # Main statistical validation experiment
    python experiments/run_single_seed.py \
        --preprocessed-dir data/ModelNet40/preprocessed \
        --hidden-dim ${HIDDEN_DIM} \
        ${USE_ASCENDER} \
        --seed ${SEED} \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3 \
        --output-dir results/statistical_validation

    RESULT_DIR="results/statistical_validation"

elif [ "${EXP_TYPE}" = "gradient" ]; then
    # Gradient analysis experiment
    python experiments/run_gradient_analysis.py \
        --preprocessed-dir data/ModelNet40/preprocessed \
        --hidden-dim ${HIDDEN_DIM} \
        --seed ${SEED} \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3 \
        --output-dir results/gradient_analysis

    RESULT_DIR="results/gradient_analysis"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================================"
echo "✅ EXPERIMENT COMPLETE: ${EXP_NAME}"
echo "============================================================================"
echo "Duration: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo "============================================================================"

# ============================================================================
# Upload Results to S3
# ============================================================================

echo ""
echo "Uploading results to S3..."

# Upload specific result directory
if [ -d "$RESULT_DIR" ]; then
    aws s3 sync "$RESULT_DIR" "s3://${BUCKET}/$RESULT_DIR" \
        --region us-east-1 \
        --no-progress
    echo "✅ Results uploaded: s3://${BUCKET}/$RESULT_DIR"
fi

# Create completion marker
mkdir -p /tmp/completion
cat > /tmp/completion/${EXP_NAME}.json << EOF
{
  "experiment": "${EXP_NAME}",
  "type": "${EXP_TYPE}",
  "hidden_dim": ${HIDDEN_DIM},
  "seed": ${SEED},
  "use_ascender": "${USE_ASCENDER}",
  "instance_id": "$(ec2-metadata --instance-id | cut -d' ' -f2)",
  "instance_type": "$(ec2-metadata --instance-type | cut -d' ' -f2)",
  "start_time": "$START_TIME",
  "end_time": "$END_TIME",
  "duration_seconds": $DURATION,
  "timestamp": "$(date -Iseconds)",
  "status": "COMPLETE"
}
EOF

aws s3 cp "/tmp/completion/${EXP_NAME}.json" \
    "s3://${BUCKET}/results/completion_markers/${EXP_NAME}.json" \
    --region us-east-1

echo "✅ Completion marker uploaded"

# ============================================================================
# Cleanup and Shutdown
# ============================================================================

echo ""
echo "Experiment complete. Instance will shutdown in 2 minutes."
echo "Check results at: s3://${BUCKET}/results/"
echo ""

# Shutdown instance to save cost
sudo shutdown -h +2
