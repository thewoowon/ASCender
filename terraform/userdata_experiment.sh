#!/bin/bash
#
# GPU Instance User Data for ASCender Experiments
# Runs a single experiment and uploads results to S3
#

set -e

BUCKET="${BUCKET}"
EXP_NAME="${EXP_NAME}"
HIDDEN_DIM="${HIDDEN_DIM}"
USE_ASCENDER="${USE_ASCENDER}"
SEED="${SEED}"
EXP_TYPE="${EXP_TYPE}"

echo "============================================================================"
echo "ASCender Experiment Runner"
echo "============================================================================"
echo "Experiment: $EXP_NAME"
echo "Type: $EXP_TYPE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "ASCender: ${USE_ASCENDER:-false}"
echo "Seed: $SEED"
echo "S3 Bucket: $BUCKET"
echo "============================================================================"

# ============================================================================
# System Setup
# ============================================================================

cd /home/ubuntu
export HOME=/home/ubuntu

# Verify GPU
nvidia-smi || { echo "ERROR: No GPU detected!"; exit 1; }

# ============================================================================
# Download Code and Data from S3
# ============================================================================

echo ""
echo "Downloading code and data from S3..."

mkdir -p /home/ubuntu/ascender
cd /home/ubuntu/ascender

# Download preprocessed data
echo "Downloading ModelNet40 preprocessed data..."
aws s3 sync "s3://$BUCKET/data/ModelNet40/preprocessed/" data/ModelNet40/preprocessed/ \
    --region us-east-1 \
    --quiet

# Download source code
echo "Downloading source code..."
aws s3 sync "s3://$BUCKET/src/" src/ --region us-east-1 --quiet
aws s3 sync "s3://$BUCKET/experiments/" experiments/ --region us-east-1 --quiet

echo "✅ Downloaded code and data"

# ============================================================================
# Setup Python Environment
# ============================================================================

echo ""
echo "Setting up Python environment..."

# Activate conda (DLAMI has conda pre-installed)
source /opt/conda/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true

# Create environment
conda create -y -n asc python=3.10 || true
conda activate asc

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy scipy tqdm matplotlib

echo "✅ Environment ready"

# Verify CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ============================================================================
# Run Experiment
# ============================================================================

echo ""
echo "============================================================================"
echo "Running experiment: $EXP_NAME"
echo "============================================================================"

cd /home/ubuntu/ascender

if [ "$EXP_TYPE" = "main" ]; then
    # Main experiment
    echo "Running main experiment..."
    python experiments/run_single_seed.py \
        --preprocessed-dir data/ModelNet40/preprocessed \
        --hidden-dim $HIDDEN_DIM \
        $USE_ASCENDER \
        --seed $SEED \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3 \
        --output-dir results/statistical_validation

    RESULT_FILE="results/statistical_validation/h${HIDDEN_DIM}_$(echo $USE_ASCENDER | grep -q 'ascender' && echo 'ascender' || echo 'baseline')_seed${SEED}.json"

elif [ "$EXP_TYPE" = "gradient" ]; then
    # Gradient analysis experiment
    echo "Running gradient analysis..."
    python experiments/run_gradient_analysis.py \
        --preprocessed-dir data/ModelNet40/preprocessed \
        --hidden-dim $HIDDEN_DIM \
        --seed $SEED \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3 \
        --output-dir results/gradient_analysis

    RESULT_FILE="results/gradient_analysis/gradient_analysis_h${HIDDEN_DIM}_seed${SEED}.json"
fi

# ============================================================================
# Upload Results to S3
# ============================================================================

echo ""
echo "Uploading results to S3..."

if [ -f "$RESULT_FILE" ]; then
    aws s3 cp "$RESULT_FILE" "s3://$BUCKET/$RESULT_FILE" --region us-east-1
    echo "✅ Results uploaded: s3://$BUCKET/$RESULT_FILE"
else
    echo "⚠️  Result file not found: $RESULT_FILE"
fi

# Upload all results directory (includes training curves)
aws s3 sync results/ "s3://$BUCKET/results/" --region us-east-1 --quiet

# ============================================================================
# Create completion marker
# ============================================================================

echo "EXPERIMENT COMPLETE: $EXP_NAME" > /tmp/experiment_complete.txt
echo "Timestamp: $(date)" >> /tmp/experiment_complete.txt
aws s3 cp /tmp/experiment_complete.txt "s3://$BUCKET/results/completion_markers/${EXP_NAME}.txt" \
    --region us-east-1

echo ""
echo "============================================================================"
echo "✅ Experiment $EXP_NAME COMPLETE!"
echo "============================================================================"
echo ""
echo "Results uploaded to: s3://$BUCKET/results/"
echo ""
echo "Instance will remain running for debugging."
echo "To terminate, use AWS Console or 'terraform destroy'"
echo ""

# Keep instance running for debugging (comment out for auto-shutdown)
# sudo shutdown -h +5  # Shutdown in 5 minutes
