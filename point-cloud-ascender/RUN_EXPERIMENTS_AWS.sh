#!/bin/bash
#
# 🚀 AWS 병렬 실험 원클릭 실행 스크립트
#
# 이 스크립트는 자동으로:
# 1. 전처리된 데이터를 S3에 업로드
# 2. 12개의 실험을 각각 별도 GPU 인스턴스에서 병렬 실행
# 3. 결과를 S3에 자동 수집
#
# 예상 시간: 2-3시간
# 예상 비용: ~$19 (g4dn.xlarge × 12 × 3시간)
#
# Usage:
#   bash RUN_EXPERIMENTS_AWS.sh

set -e

# ============================================================================
# 설정
# ============================================================================

# S3 버킷 이름 (자동 생성됨)
S3_BUCKET="ascender-experiments-$(date +%s)"
S3_PATH="s3://${S3_BUCKET}"

# AWS 리전
AWS_REGION="us-east-1"

# EC2 설정
INSTANCE_TYPE="g4dn.xlarge"  # 1x NVIDIA T4 GPU, $0.526/hr
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS with CUDA 11.8

# 실험 설정
EPOCHS=50
BATCH_SIZE=32

echo ""
echo "================================================================================"
echo "🚀 ASCender Statistical Validation - AWS Parallel Execution"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Region: ${AWS_REGION}"
echo "  - Instance Type: ${INSTANCE_TYPE}"
echo "  - Number of Instances: 12 (parallel execution)"
echo "  - Epochs per experiment: ${EPOCHS}"
echo "  - S3 Bucket: ${S3_BUCKET}"
echo ""
echo "Cost Estimate:"
echo "  - Instance cost: \$0.526/hr × 12 instances × 3 hours = ~\$18.94"
echo "  - S3 storage: ~\$0.10"
echo "  - Total: ~\$19"
echo ""
echo "Timeline:"
echo "  - Setup & data upload: ~5 min"
echo "  - Training: ~2-3 hours (parallel)"
echo "  - Results download: ~2 min"
echo "  - Total: ~2.5-3.5 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user."
    exit 1
fi

# ============================================================================
# Step 1: S3 버킷 생성 및 데이터 업로드
# ============================================================================

echo ""
echo "================================================================================"
echo "Step 1/5: Creating S3 bucket and uploading data..."
echo "================================================================================"

# S3 버킷 생성
aws s3 mb ${S3_PATH} --region ${AWS_REGION}
echo "✅ S3 bucket created: ${S3_PATH}"

# 전처리된 데이터 업로드
echo "Uploading preprocessed ModelNet40 data..."
aws s3 sync data/ModelNet40/preprocessed/ \
    ${S3_PATH}/data/ModelNet40/preprocessed/ \
    --region ${AWS_REGION} \
    --quiet

# 소스 코드 업로드
echo "Uploading source code..."
aws s3 sync src/ ${S3_PATH}/src/ --region ${AWS_REGION} --quiet
aws s3 sync experiments/ ${S3_PATH}/experiments/ --region ${AWS_REGION} --quiet

echo "✅ Data and code uploaded to S3"

# ============================================================================
# Step 2: 보안 그룹 생성 (이미 있으면 재사용)
# ============================================================================

echo ""
echo "================================================================================"
echo "Step 2/5: Setting up security group..."
echo "================================================================================"

# 기본 VPC 가져오기
DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text \
    --region ${AWS_REGION})

if [ "${DEFAULT_VPC}" == "None" ]; then
    echo "❌ Error: No default VPC found. Please create a VPC first."
    exit 1
fi

echo "Using default VPC: ${DEFAULT_VPC}"

# 보안 그룹 생성 또는 재사용
SG_NAME="ascender-sg"
SECURITY_GROUP=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region ${AWS_REGION} 2>/dev/null || echo "None")

if [ "${SECURITY_GROUP}" == "None" ]; then
    echo "Creating new security group..."
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --group-name ${SG_NAME} \
        --description "ASCender experiments security group" \
        --vpc-id ${DEFAULT_VPC} \
        --region ${AWS_REGION} \
        --query 'GroupId' \
        --output text)

    # SSH 접속 허용 (필요시)
    aws ec2 authorize-security-group-ingress \
        --group-id ${SECURITY_GROUP} \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region ${AWS_REGION} || true

    echo "✅ Security group created: ${SECURITY_GROUP}"
else
    echo "✅ Using existing security group: ${SECURITY_GROUP}"
fi

# ============================================================================
# Step 3: 12개 EC2 인스턴스 실행 (병렬)
# ============================================================================

echo ""
echo "================================================================================"
echo "Step 3/5: Launching 12 GPU instances (parallel)..."
echo "================================================================================"

# 실험 목록
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

declare -a INSTANCE_IDS=()

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name hidden_dim use_ascender seed <<< "${exp}"

    echo "Launching: ${exp_name}..."

    # User data script
    USER_DATA=$(cat <<EOF
#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log) 2>&1

# Install dependencies
apt-get update
apt-get install -y python3-pip python3-venv awscli

# Create working directory
mkdir -p /home/ubuntu/ascender
cd /home/ubuntu/ascender

# Download data and code from S3
aws s3 sync ${S3_PATH}/data/ data/ --region ${AWS_REGION}
aws s3 sync ${S3_PATH}/src/ src/ --region ${AWS_REGION}
aws s3 sync ${S3_PATH}/experiments/ experiments/ --region ${AWS_REGION}

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy tqdm

# Run experiment
python experiments/run_single_seed.py \\
    --hidden-dim ${hidden_dim} \\
    --seed ${seed} \\
    $( [ "${use_ascender}" == "1" ] && echo "--use-ascender" || echo "" ) \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --preprocessed-dir data/ModelNet40/preprocessed \\
    --output-dir results/statistical_validation

# Upload results to S3
aws s3 cp results/statistical_validation/ ${S3_PATH}/results/statistical_validation/ --recursive --region ${AWS_REGION}

# Shutdown instance after completion
shutdown -h now
EOF
)

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --region ${AWS_REGION} \
        --image-id ${AMI_ID} \
        --instance-type ${INSTANCE_TYPE} \
        --security-group-ids ${SECURITY_GROUP} \
        --user-data "${USER_DATA}" \
        --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ascender-${exp_name}},{Key=Project,Value=ASCender}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    INSTANCE_IDS+=("${INSTANCE_ID}")
    echo "  ✅ ${INSTANCE_ID}"

    sleep 1  # Avoid rate limiting
done

# Save instance IDs
echo "${INSTANCE_IDS[@]}" > /tmp/ascender_instance_ids.txt

echo ""
echo "✅ All 12 instances launched successfully!"
echo ""
echo "Instance IDs:"
for id in "${INSTANCE_IDS[@]}"; do
    echo "  - ${id}"
done

# ============================================================================
# Step 4: 진행 상황 모니터링
# ============================================================================

echo ""
echo "================================================================================"
echo "Step 4/5: Monitoring experiment progress..."
echo "================================================================================"
echo ""
echo "Experiments are now running on AWS EC2 instances."
echo ""
echo "Timeline:"
echo "  - Setup & installation: ~5-10 min"
echo "  - Training (50 epochs): ~2-3 hours"
echo "  - Results upload: ~1 min"
echo "  - Auto-shutdown: automatic"
echo ""
echo "You can monitor progress by checking S3 for result files:"
echo "  aws s3 ls ${S3_PATH}/results/statistical_validation/"
echo ""
echo "Or check instance status:"
echo "  aws ec2 describe-instances --instance-ids ${INSTANCE_IDS[0]} --query 'Reservations[0].Instances[0].State.Name'"
echo ""

# Wait for results
echo "Waiting for experiments to complete..."
echo "(This will take approximately 2-3 hours. You can safely close this terminal.)"
echo ""

COMPLETED=0
TOTAL=12

while [ ${COMPLETED} -lt ${TOTAL} ]; do
    # Count result files in S3
    COMPLETED=$(aws s3 ls ${S3_PATH}/results/statistical_validation/ 2>/dev/null | grep '.json' | wc -l)

    echo -ne "Progress: ${COMPLETED}/${TOTAL} experiments completed\r"
    sleep 60  # Check every minute
done

echo ""
echo "✅ All experiments completed!"

# ============================================================================
# Step 5: 결과 다운로드 및 분석
# ============================================================================

echo ""
echo "================================================================================"
echo "Step 5/5: Downloading results and computing statistics..."
echo "================================================================================"

# Download results
mkdir -p results/statistical_validation
aws s3 sync ${S3_PATH}/results/statistical_validation/ \
    results/statistical_validation/ \
    --region ${AWS_REGION}

echo "✅ Results downloaded"

# Compute statistics
python experiments/compute_statistics.py

echo ""
echo "================================================================================"
echo "🎉 SUCCESS! Statistical validation complete!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - results/statistical_validation/summary.json"
echo "  - results/statistical_validation/summary_table.txt"
echo ""
echo "Next steps:"
echo "  1. Review results: cat results/statistical_validation/summary_table.txt"
echo "  2. Update papers with statistical results"
echo "  3. Clean up AWS resources: aws s3 rb ${S3_PATH} --force"
echo ""
echo "Total cost: Check AWS billing console for final charges (~\$19 estimated)"
echo "================================================================================"
