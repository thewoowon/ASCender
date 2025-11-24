# ASCender v2.0 - Quick Start Guide

## 🎯 TL;DR (가장 빠른 방법)

**기존 Terraform 인프라를 활용한 원클릭 실행**:

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# 1. GPU 실험 시작 (자동으로 코드 업로드 + 인스턴스 시작)
bash run_gpu_experiment.sh

# 2. SSH 접속 (출력된 IP 사용)
ssh -i ~/.ssh/ascender-key.pem ubuntu@<PUBLIC_IP>

# 3. 실험 실행
cd /home/ubuntu/asc/code/point-cloud-ascender
conda activate asc
pip install scikit-learn h5py
python experiments/modelnet40_gpu_experiment.py

# 4. 완료 후 중단
bash stop_gpu_experiment.sh
```

**예상 비용**: ~$0.60 (g5.xlarge spot, 2시간)

---

## 📚 What's Inside

### 🐛 Bug Fixes (v2.0)

4가지 치명적인 버그를 수정했습니다:

1. ✅ **p_r overwrite bug** - ASC가 이제 실제 좌표 사용
2. ✅ **Graph reweighting** - Level 1 활성화
3. ✅ **Surface normals** - PCA 기반 실제 normal 계산
4. ✅ **Augmentation** - Rotation/scaling/jitter 추가

자세한 내용: [GPU_EXPERIMENT_GUIDE.md](./GPU_EXPERIMENT_GUIDE.md)

### 🗂️ File Structure

```
point-cloud-ascender/
├── QUICKSTART.md                  ← 여기!
├── GPU_EXPERIMENT_GUIDE.md        ← 상세 가이드 (bug fixes 설명)
├── TERRAFORM_GPU_GUIDE.md         ← Terraform 활용법
│
├── run_gpu_experiment.sh          ← 원클릭 실행
├── stop_gpu_experiment.sh         ← 원클릭 정리
│
├── experiments/
│   ├── modelnet40_experiment.py        ← CPU 버전 (테스트용)
│   └── modelnet40_gpu_experiment.py    ← GPU 버전 (실제 실험)
│
└── src/models/
    └── point_ascender_v2.py       ← 버그 수정된 코어 모델
```

---

## 🚀 Usage Options

### Option A: Terraform (Recommended) - 완전 자동화

**장점**: 자동 설정, Spot 보호, S3 자동 sync

```bash
# 원클릭 실행
bash run_gpu_experiment.sh

# 인스턴스에서 실험 실행 (위 TL;DR 참고)

# 완료 후 정리
bash stop_gpu_experiment.sh
```

**상세 가이드**: [TERRAFORM_GPU_GUIDE.md](./TERRAFORM_GPU_GUIDE.md)

### Option B: Manual AWS Launch - 수동 제어

**장점**: 세밀한 제어, 다른 인스턴스 타입 쉽게 변경

```bash
# 1. 스크립트로 수동 시작
export AWS_KEY_NAME=your-key-name
export AWS_SG_ID=sg-xxxxxxxxxxxxx
bash launch_aws_gpu.sh

# 2. 연결 및 설정
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}
bash setup_gpu_instance.sh

# 3. 코드 업로드 (로컬에서)
scp -r point-cloud-ascender ubuntu@${PUBLIC_IP}:~/ASCender/

# 4. 실험 실행 (인스턴스에서)
cd ~/ASCender/point-cloud-ascender
python3 experiments/modelnet40_gpu_experiment.py

# 5. 정리
bash terminate_aws_gpu.sh
```

**상세 가이드**: [GPU_EXPERIMENT_GUIDE.md](./GPU_EXPERIMENT_GUIDE.md)

### Option C: Local CPU (테스트용)

**장점**: 비용 없음, 빠른 디버깅

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# 간단한 CPU 실험 (50 epochs, 1 layer)
python experiments/modelnet40_experiment.py
```

**주의**: CPU는 너무 느려서 실제 실험에는 부적합 (~2-3시간)

---

## 📊 Expected Results

버그 수정 후 예상 결과:

| Model Size | Params | Baseline | ASCender | Improvement | α |
|------------|--------|----------|----------|-------------|---|
| Small (50K) | 50K | 60% | **67%** | **+7%** ✅ | ~0.3 |
| Medium (200K) | 200K | 68% | **70%** | **+2%** ✅ | ~0.5 |
| Large (500K) | 500K | 72% | 72% | ±0% | ~0.7 |

**핵심 개선**:
- ✅ Small 모델에서 큰 향상 (+7%)
- ✅ α가 낮아져서 bias 실제로 활용 (~0.3-0.5)
- ✅ α saturation 해결 (이전: 0.97 → 지금: 0.3-0.5)

---

## 💰 Cost Comparison

| Method | Instance | 실행 시간 | 비용 | 특징 |
|--------|----------|---------|------|-----|
| **Terraform (Spot)** | g5.xlarge spot | ~2h | **$0.60** | ✅ 추천 |
| Manual (Spot) | g4dn.xlarge spot | ~2.5h | $0.40 | 조금 느림 |
| Manual (On-demand) | g5.xlarge | ~2h | $2.01 | 안정적 |
| Local CPU | MacBook | ~3-5h | $0 | 테스트용 |

**Recommendation**: Terraform + Spot = **최고의 가성비**

---

## 🔧 Troubleshooting

### 실험이 시작되지 않음

```bash
# 1. Terraform 상태 확인
cd /Users/aepeul/ASCender/terraform
terraform show

# 2. 인스턴스 확인
aws ec2 describe-instances \
    --filters "Name=tag:aws:autoscaling:groupName,Values=asc-gpu-asg" \
    --query "Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]"

# 3. Auto Scaling Group 활동 확인
aws autoscaling describe-scaling-activities \
    --auto-scaling-group-name asc-gpu-asg \
    --max-records 3
```

### SSH 연결 안됨

```bash
# Security group에 내 IP 추가
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id sg-0a670dda5d7fec182 \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32
```

### GPU 메모리 부족 (OOM)

`modelnet40_gpu_experiment.py`에서 batch size 줄이기:

```python
train_loader = DataLoader(..., batch_size=16)  # Was 32
```

### 실험 중간에 중단됨 (Spot)

걱정 마세요! `asc-spot-guard` 서비스가:
- 자동으로 체크포인트를 S3에 업로드
- 로그도 함께 저장

재시작 후 체크포인트에서 이어서 실행 가능.

---

## 📖 Detailed Guides

각 상황별 상세 가이드:

1. **버그 수정 내용 + 코드 설명**
   → [GPU_EXPERIMENT_GUIDE.md](./GPU_EXPERIMENT_GUIDE.md)

2. **Terraform 활용법 + 고급 설정**
   → [TERRAFORM_GPU_GUIDE.md](./TERRAFORM_GPU_GUIDE.md)

3. **AWS 수동 설정 + 인스턴스 선택**
   → [aws_gpu_setup.md](./aws_gpu_setup.md)

---

## 🎓 Next Steps

실험 완료 후:

1. **결과 분석**
   ```bash
   # 결과 확인
   cat results/modelnet40_gpu_results.json

   # α evolution 확인
   grep "α" logs/gpu_experiment.log
   ```

2. **논문 업데이트**
   - Section 4.5에 ModelNet40 결과 추가
   - Abstract에 real-world validation 언급
   - Methods에 preprocessing 상세 설명

3. **추가 실험 (Optional)**
   - 다른 데이터셋 (ShapeNet, ScanNet)
   - Ablation: 각 fix의 개별 효과
   - Hyperparameter tuning (w_align, w_sep, w_coh)

---

## 📞 Questions?

- **빠른 시작**: 위의 TL;DR 따라하기
- **Terraform 사용법**: [TERRAFORM_GPU_GUIDE.md](./TERRAFORM_GPU_GUIDE.md)
- **버그 수정 상세**: [GPU_EXPERIMENT_GUIDE.md](./GPU_EXPERIMENT_GUIDE.md)
- **AWS 수동 설정**: [aws_gpu_setup.md](./aws_gpu_setup.md)

**가장 쉬운 방법**: `bash run_gpu_experiment.sh` 실행! 🚀
