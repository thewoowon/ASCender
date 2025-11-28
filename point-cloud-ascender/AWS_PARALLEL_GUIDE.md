# AWS 병렬 실행 가이드

## 🚀 빠른 시작 (Quick Start)

### 필수 조건
1. AWS CLI 설치 및 설정
2. S3 버킷 생성
3. EC2 SSH 키 페어 생성

### 실행 단계

#### 1️⃣ AWS 설정 편집

`aws_parallel_launcher.sh` 파일을 열어서 다음 값들을 수정:

```bash
AWS_REGION="us-east-1"                    # 원하는 리전
INSTANCE_TYPE="g4dn.xlarge"               # GPU 인스턴스 타입
AMI_ID="ami-0c7217cdde317cfec"            # Ubuntu 22.04 CUDA AMI
KEY_NAME="your-key-name"                  # SSH 키 이름
SECURITY_GROUP="sg-xxxxxxxx"              # 보안 그룹 ID
SUBNET_ID="subnet-xxxxxxxx"               # 서브넷 ID
S3_BUCKET="s3://your-bucket/ascender"     # S3 버킷 경로
```

#### 2️⃣ 병렬 실험 시작

```bash
chmod +x aws_parallel_launcher.sh
bash aws_parallel_launcher.sh
```

이 스크립트는 자동으로:
- ✅ 전처리된 데이터를 S3에 업로드 (~5분)
- ✅ 12개의 GPU 인스턴스 실행
- ✅ 각 인스턴스에서 독립적으로 학습 시작

#### 3️⃣ 결과 다운로드

실험이 완료되면 (약 2-3시간 후):

```bash
chmod +x aws_download_results.sh
bash aws_download_results.sh
```

#### 4️⃣ 통계 분석

```bash
python experiments/compute_statistics.py
```

결과는 다음 위치에 저장됩니다:
- `results/statistical_validation/summary.json`
- `results/statistical_validation/summary_table.txt`

---

## 📊 예상 비용 및 시간

### 인스턴스 타입별 비교

| 인스턴스 타입 | GPU | 시간당 비용 | 실험 시간 | 총 비용 (12개) |
|--------------|-----|------------|----------|---------------|
| **g4dn.xlarge** | 1x T4 (16GB) | $0.526 | ~3시간 | **~$19** |
| g4dn.2xlarge | 1x T4 (16GB) | $0.752 | ~2시간 | ~$18 |
| p3.2xlarge | 1x V100 (16GB) | $3.06 | ~1.5시간 | ~$55 |

**추천:** `g4dn.xlarge` (가성비 최고, 충분히 빠름)

### 병렬화 옵션

#### 옵션 A: 최대 속도 (12개 인스턴스)
- **시간:** 2-3시간
- **비용:** ~$19
- **장점:** 가장 빠름, 내일 제출에 여유

#### 옵션 B: 절약형 (3개 인스턴스, 순차 실행)
- **시간:** 8-12시간
- **비용:** ~$5
- **장점:** 저렴, 밤에 돌리면 아침에 완료

#### 옵션 C: 중간형 (6개 인스턴스)
- **시간:** 4-6시간
- **비용:** ~$10
- **장점:** 비용과 시간 밸런스

---

## 🔧 상세 설정

### 1. AWS CLI 설정

```bash
# AWS CLI 설치 (Mac)
brew install awscli

# AWS CLI 설정
aws configure
# AWS Access Key ID: [입력]
# AWS Secret Access Key: [입력]
# Default region: us-east-1
# Default output format: json
```

### 2. S3 버킷 생성

```bash
# S3 버킷 생성
aws s3 mb s3://your-bucket-name --region us-east-1

# 버킷 확인
aws s3 ls
```

### 3. EC2 SSH 키 페어 생성

```bash
# 키 페어 생성
aws ec2 create-key-pair \
    --key-name ascender-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/ascender-key.pem

# 권한 설정
chmod 400 ~/.ssh/ascender-key.pem

# 키 이름 확인
aws ec2 describe-key-pairs
```

### 4. 보안 그룹 및 VPC 설정

```bash
# 기본 VPC ID 확인
aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text

# 기본 서브넷 ID 확인
aws ec2 describe-subnets --query 'Subnets[0].SubnetId' --output text

# 보안 그룹 생성 (SSH 허용)
aws ec2 create-security-group \
    --group-name ascender-sg \
    --description "ASCender experiments security group"

# SSH 접속 허용
aws ec2 authorize-security-group-ingress \
    --group-name ascender-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

---

## 📝 실행 예제

### 전체 자동화 실행

```bash
# 1. 설정 파일 편집
nano aws_parallel_launcher.sh
# (AWS_REGION, S3_BUCKET 등 수정)

# 2. 병렬 실험 시작
bash aws_parallel_launcher.sh

# 출력:
# ============================================================================
# AWS Parallel Launcher for Statistical Validation
# ============================================================================
#
# This will launch 12 g4dn.xlarge instances
# Estimated cost: $18.936 for 3 hours
#
# Continue? (y/n)

# 3. 진행 상황 모니터링
watch -n 60 'aws s3 ls s3://your-bucket/ascender/results/statistical_validation/'

# 4. 결과 다운로드 (3시간 후)
bash aws_download_results.sh

# 5. 통계 분석
python experiments/compute_statistics.py
```

---

## 🎯 단일 실험 테스트

본격적인 병렬 실행 전에 단일 실험으로 테스트:

```bash
# 로컬에서 테스트 (CPU)
python experiments/run_single_seed.py \
    --hidden-dim 32 \
    --seed 42 \
    --epochs 5 \
    --batch-size 16

# AWS에서 단일 인스턴스 테스트
# (aws_parallel_launcher.sh에서 NUM_INSTANCES=1로 설정)
```

---

## 📊 모니터링

### 인스턴스 상태 확인

```bash
# 실행 중인 인스턴스 확인
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=ASCender-StatValidation" \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
    --output table

# 특정 인스턴스 로그 확인 (SSH 접속)
INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids i-xxxxxxxxx \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

ssh -i ~/.ssh/ascender-key.pem ubuntu@${INSTANCE_IP}
# 인스턴스 내부에서:
tail -f ~/ascender/experiment.log
```

### S3 결과 확인

```bash
# 업로드된 결과 파일 확인
aws s3 ls s3://your-bucket/ascender/results/statistical_validation/

# 개별 결과 미리보기
aws s3 cp s3://your-bucket/ascender/results/statistical_validation/h32_baseline_seed42.json - | jq .
```

---

## 🛠️ 문제 해결

### 문제 1: "No default VPC" 에러

```bash
# 새 VPC 생성
aws ec2 create-vpc --cidr-block 10.0.0.0/16
aws ec2 create-subnet --vpc-id vpc-xxxxx --cidr-block 10.0.1.0/24
```

### 문제 2: 인스턴스 실행 권한 없음

```bash
# IAM 권한 확인
aws iam get-user

# EC2 실행 권한 필요:
# - ec2:RunInstances
# - ec2:DescribeInstances
# - s3:PutObject
# - s3:GetObject
```

### 문제 3: GPU Out of Memory

```bash
# batch-size 줄이기
python experiments/run_single_seed.py \
    --batch-size 16  # 기본 32 → 16으로 감소
```

### 문제 4: 비용 초과 방지

```bash
# 실행 중인 모든 인스턴스 중지
aws ec2 stop-instances --instance-ids $(cat /tmp/ascender_instance_ids.txt)

# 인스턴스 삭제
aws ec2 terminate-instances --instance-ids $(cat /tmp/ascender_instance_ids.txt)
```

---

## 💰 비용 최적화 팁

1. **Spot Instances 사용** - 70% 저렴 (단, 중단 가능)
2. **최소 인스턴스로 시작** - h32만 먼저 (6개 인스턴스)
3. **야간 실행** - 낮에는 비용이 높을 수 있음
4. **자동 종료 설정** - 실험 완료 후 자동 종료

```bash
# Spot Instance로 실행 (aws_parallel_launcher.sh 수정)
--instance-market-options "MarketType=spot,SpotOptions={MaxPrice=0.3,SpotInstanceType=one-time}"
```

---

## ✅ 체크리스트

실행 전 확인사항:

- [ ] AWS CLI 설치 및 설정 완료
- [ ] S3 버킷 생성 완료
- [ ] EC2 키 페어 생성 완료
- [ ] 보안 그룹 및 VPC 설정 완료
- [ ] `aws_parallel_launcher.sh` 설정 완료
- [ ] 전처리된 데이터 확인 (`data/ModelNet40/preprocessed/`)
- [ ] 예산 확인 (~$19 for 12 instances)

실행 후 확인사항:

- [ ] 12개 인스턴스 모두 실행 확인
- [ ] S3에 데이터 업로드 확인
- [ ] 2-3시간 후 결과 파일 12개 확인
- [ ] 통계 분석 완료
- [ ] 논문에 결과 반영

---

## 📧 지원

문제가 발생하면:
1. AWS Console에서 CloudWatch 로그 확인
2. 인스턴스에 SSH 접속하여 로그 확인
3. S3 버킷에 업로드된 파일 확인

**긴급 중단:**
```bash
# 모든 실험 즉시 중단
aws ec2 terminate-instances --instance-ids $(cat /tmp/ascender_instance_ids.txt)
```
