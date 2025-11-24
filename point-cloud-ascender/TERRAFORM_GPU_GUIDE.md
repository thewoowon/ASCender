# ASCender GPU Experiment - Terraform Quick Start

## 🎯 Overview

기존 Terraform 설정을 활용하여 **자동으로** GPU 인스턴스를 띄우고 실험을 실행합니다.

**장점**:
- ✅ **완전 자동화**: userdata 스크립트로 환경 자동 설정
- ✅ **Spot instance**: capacity-optimized로 70% 저렴
- ✅ **Spot 보호**: 중단 시 자동으로 체크포인트 업로드
- ✅ **S3 통합**: 코드/데이터/결과 자동 sync
- ✅ **이미 검증됨**: 기존 설정 활용

## 📋 Prerequisites

```bash
# Terraform 설치 확인
terraform version  # >= 1.5.0

# AWS credentials 확인
cd /Users/aepeul/ASCender/terraform
terraform validate
```

## 🚀 Quick Start (3 Steps)

### Step 1: 코드 업로드 to S3

```bash
cd /Users/aepeul/ASCender

# 수정된 코드를 S3에 업로드
aws s3 sync point-cloud-ascender/ s3://ascender-research-20251005/code/point-cloud-ascender/ \
    --exclude ".git/*" \
    --exclude "__pycache__/*" \
    --exclude "*.pyc"

# ModelNet40 데이터도 업로드 (이미 다운로드 되어있다면)
aws s3 sync point-cloud-ascender/data/modelnet40_ply_hdf5_2048/ \
    s3://ascender-research-20251005/data/modelnet40/ \
    --exclude "*.txt"
```

### Step 2: GPU 인스턴스 시작

```bash
cd /Users/aepeul/ASCender/terraform

# 현재 상태 확인
terraform plan

# GPU Auto Scaling Group 시작 (desired_capacity = 1로 설정됨)
terraform apply -auto-approve

# 인스턴스 정보 확인
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=asc-gpu-*" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].[InstanceId,PublicIpAddress,InstanceType]" \
    --output table
```

**출력 예시**:
```
---------------------------------------------------------
|              DescribeInstances                       |
+-----------------------+----------------+--------------+
|  i-0abc123def456789   | 3.89.45.123    | g5.xlarge   |
+-----------------------+----------------+--------------+
```

### Step 3: 연결 및 실험 실행

```bash
# 변수 저장
INSTANCE_IP="3.89.45.123"  # 위에서 확인한 IP

# SSH 연결
ssh -i ~/.ssh/ascender-key.pem ubuntu@$INSTANCE_IP

# 인스턴스에서 실행:
cd /home/ubuntu/asc/code/point-cloud-ascender

# ModelNet40 데이터 확인
ls -lh data/modelnet40/

# GPU 확인
nvidia-smi

# 실험 실행 (tmux 사용 권장)
tmux new -s modelnet40
conda activate asc  # userdata에서 이미 생성됨

# Point cloud 전용 패키지 설치
pip install scikit-learn h5py

# 실험 실행
python experiments/modelnet40_gpu_experiment.py 2>&1 | tee logs/gpu_experiment.log

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t modelnet40
```

## 📊 모니터링

### GPU 사용률 모니터링

```bash
# 별도 세션에서
watch -n 1 nvidia-smi
```

### 실험 진행 상황

```bash
# 로그 tail
tail -f logs/gpu_experiment.log

# α 값 확인
grep "α" logs/gpu_experiment.log

# 현재 epoch 확인
grep "Epoch" logs/gpu_experiment.log | tail -5
```

### Spot Interruption 모니터링

Spot 인스턴스가 중단될 예정이면:
- `asc-spot-guard` 서비스가 자동으로 체크포인트를 S3에 업로드
- 로그 확인: `sudo journalctl -u asc-spot-guard -f`

## 💾 결과 다운로드

### 실험 완료 후 (인스턴스에서)

```bash
# 결과를 S3에 업로드
aws s3 sync results/ s3://ascender-research-20251005/results/modelnet40/ \
    --exclude "*.pyc"

aws s3 sync logs/ s3://ascender-research-20251005/logs/modelnet40/ \
    --include "*.log"
```

### 로컬에서 다운로드

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# 결과 다운로드
aws s3 sync s3://ascender-research-20251005/results/modelnet40/ results/

# 로그 다운로드
aws s3 sync s3://ascender-research-20251005/logs/modelnet40/ logs/

# 확인
cat results/modelnet40_gpu_results.json
```

## 🛑 정리 (Cleanup)

### 실험 완료 후 인스턴스 종료

```bash
cd /Users/aepeul/ASCender/terraform

# Auto Scaling Group의 desired capacity를 0으로
terraform apply -var="gpu_asg_desired_capacity=0"

# 또는 완전히 destroy (주의!)
terraform destroy -target=aws_autoscaling_group.gpu_asg
```

**중요**: 비용 절감을 위해 **반드시** 종료하세요!

## 📝 Terraform 설정 활용

### 현재 구성 요약

기존 `main.tf`에 이미 설정된 것들:

1. **S3 Bucket**: `ascender-research-20251005`
   - Lifecycle: 30일 후 IA, 90일 후 Glacier
   - 코드/데이터/결과 저장

2. **IAM Role/Profile**: `AscenderEC2Profile`
   - S3 full access
   - CloudWatch Logs access

3. **GPU Launch Template**:
   - Image: Deep Learning AMI (PyTorch, CUDA 12.1)
   - Instance: g5.xlarge (primary), g4dn.xlarge (fallback)
   - Storage: 200GB gp3
   - Network: Public IP, 기존 security group

4. **Auto Scaling Group**:
   - Spot allocation: capacity-optimized
   - Min: 0, Max: 1, Desired: 1
   - First instance: on-demand (안정성)

5. **Userdata Script** (`userdata_gpu.sh`):
   - Conda 환경 자동 생성
   - PyTorch + CUDA 설치
   - S3에서 코드/데이터 sync
   - Spot guard 서비스 시작

### Point Cloud 실험용 커스터마이즈

기존 userdata를 point cloud용으로 확장하려면:

```bash
cd /Users/aepeul/ASCender/terraform

# userdata_gpu.sh 수정
cat >> userdata_gpu.sh <<'EOF'

# Point cloud specific packages
conda activate asc
pip install scikit-learn h5py open3d

# Download ModelNet40 if not in S3
if [ ! -d "/home/ubuntu/asc/data/modelnet40" ]; then
  echo "Downloading ModelNet40..."
  cd /home/ubuntu/asc/data
  wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
  unzip modelnet40_ply_hdf5_2048.zip
  rm modelnet40_ply_hdf5_2048.zip
fi

# Auto-start experiment (optional)
# cd /home/ubuntu/asc/code/point-cloud-ascender
# tmux new -d -s auto "python experiments/modelnet40_gpu_experiment.py"
EOF

# Apply changes
terraform apply -auto-approve
```

## 💰 비용 추정

| 리소스 | 타입 | 시간당 | 2시간 실험 |
|--------|------|---------|-----------|
| GPU Instance | g5.xlarge spot | ~$0.30 | ~$0.60 |
| Orchestrator | c7i.large on-demand | $0.15 | $0.30 |
| S3 Storage | Standard | $0.023/GB | ~$0.05 |
| **Total** | | | **~$0.95** |

**참고**:
- g5.xlarge on-demand: $1.006/hr
- Spot으로 **70% 절감**!

## 🔧 Troubleshooting

### 인스턴스가 시작되지 않음

```bash
# Auto Scaling Group 활동 확인
aws autoscaling describe-scaling-activities \
    --auto-scaling-group-name asc-gpu-asg \
    --max-records 5

# Launch Template 확인
terraform state show aws_launch_template.gpu_lt
```

### userdata 실행 확인

```bash
# 인스턴스에서
sudo cat /var/log/cloud-init-output.log

# 에러 확인
sudo grep -i error /var/log/cloud-init-output.log
```

### S3 접근 권한 문제

```bash
# IAM role 확인
aws sts get-caller-identity

# S3 접근 테스트
aws s3 ls s3://ascender-research-20251005/
```

## 🎓 고급 사용법

### 여러 실험 병렬 실행

```bash
# Auto Scaling Group max_size 증가
terraform apply -var="gpu_asg_max_size=3" -var="gpu_asg_desired_capacity=3"

# 각 인스턴스에서 다른 설정 실행
# Instance 1: Small model
# Instance 2: Medium model
# Instance 3: Large model
```

### 긴 실험 자동화

userdata에 실험 자동 시작 추가:

```bash
# userdata_gpu.sh 끝에 추가
cat <<'SCRIPT' > /home/ubuntu/auto_experiment.sh
#!/bin/bash
cd /home/ubuntu/asc/code/point-cloud-ascender
conda activate asc
python experiments/modelnet40_gpu_experiment.py 2>&1 | tee logs/auto_run_$(date +%Y%m%d_%H%M%S).log
aws s3 sync results/ s3://ascender-research-20251005/results/
aws s3 sync logs/ s3://ascender-research-20251005/logs/
sudo shutdown -h now  # 완료 후 자동 종료
SCRIPT
chmod +x /home/ubuntu/auto_experiment.sh
sudo -u ubuntu tmux new -d -s auto "/home/ubuntu/auto_experiment.sh"
```

---

## 📚 Related Files

- **Terraform Config**: [/Users/aepeul/ASCender/terraform/main.tf](../../terraform/main.tf)
- **GPU Userdata**: [/Users/aepeul/ASCender/terraform/userdata_gpu.sh](../../terraform/userdata_gpu.sh)
- **Variables**: [/Users/aepeul/ASCender/terraform/terraform.tfvars](../../terraform/terraform.tfvars)
- **Experiment Script**: [experiments/modelnet40_gpu_experiment.py](../experiments/modelnet40_gpu_experiment.py)

**Questions?** 기존 Terraform 설정이 이미 완전히 구성되어 있으므로, 위 3단계만 따라하면 됩니다!
