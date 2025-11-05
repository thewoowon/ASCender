# ASCender 실험 환경 정리

> **업데이트**: 2025-11-06
> **목적**: 인프라 및 실험 설정 핵심 요약

---

## 📦 1. 인프라 구성 (Terraform)

### 1.1 아키텍처
```
┌──────────────────┐
│  Orchestrator    │  c7i.large (On-Demand)
│  - 실험 관리     │  - 스케줄링, 모니터링
│  - 결과 수집     │  - S3 sync
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  GPU Worker      │  g5.xlarge (Spot)
│  - 학습 실행     │  - 24GB VRAM (A10G)
│  - 체크포인트    │  - Capacity-optimized
└──────────────────┘
         │
         ▼
┌──────────────────┐
│  S3 Bucket       │
│  - Checkpoints   │  Lifecycle: 30d→IA, 90d→Glacier
│  - Logs/Results  │
└──────────────────┘
```

### 1.2 핵심 리소스

#### GPU Worker (g5.xlarge Spot)
- **Instance**: g5.xlarge (4 vCPU, 16GB RAM, 24GB VRAM)
- **AMI**: Deep Learning AMI (PyTorch GPU)
- **Strategy**: Spot capacity-optimized (비용 절감)
- **Storage**: 200GB gp3 EBS
- **Auto Scaling**: 0-1 (필요시만 가동)

#### Orchestrator (c7i.large On-Demand)
- **Role**: 실험 스케줄링, 로그 수집
- **Instance**: c7i.large (compute-optimized)
- **Always-on**: 안정적 관리 서버

#### S3 Storage
- **Tiering**:
  - 0-30일: Standard
  - 30-90일: Standard-IA
  - 90일+: Glacier Instant Retrieval
- **IAM**: EC2 instance profile로 접근

### 1.3 주요 변수 (terraform.tfvars 필요)
```hcl
region           = "us-east-1"
subnet_id        = "subnet-xxxxx"       # 기존 VPC의 public subnet
security_group   = "sg-xxxxx"            # SSH(22) 허용
key_name         = "your-keypair"
bucket_name      = "ascender-experiments"
gpu_ami          = "ami-xxxxxxxx"        # DLAMI (GPU, PyTorch)
cpu_ami          = "ami-xxxxxxxx"        # Amazon Linux 2023
```

### 1.4 배포 명령
```bash
cd /Users/aepeul/ASCender/terraform
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -auto-approve

# GPU 워커만 스케일 조정
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name asc-gpu-asg \
  --desired-capacity 1  # 0으로 설정하면 종료
```

---

## 🧪 2. 실험 설정

### 2.1 Config 파일 체계

#### 주요 Config
| 파일 | 목적 | 설정 |
|------|------|------|
| `baseline.yaml` | Vanilla Transformer | ASCender OFF |
| `ascender256.yaml` | **메인 실험** | Cohesion+Align, r=0.30 |
| `ascender_safe.yaml` | Conservative | r=0.25, ceiling=0.65 |
| `ascender_very_aggressive.yaml` | High-impact | r=0.35, residual path |
| `ascender256_residual.yaml` | Residual Path 테스트 | Dual-path 아키텍처 |

#### 파생 Config (특수 목적)
- `ascender256_moderate.yaml`: 중간 강도
- `ascender256_emergent.yaml`: Emergent structure 실험용
- `ascender256_residual_wt103.yaml`: WikiText-103 대용량

### 2.2 핵심 실험 파라미터

#### Dataset (WikiText-2)
```yaml
dataset:
  name: "wikitext-2-v1"
  seq_len: 256
  batch_size: 4
  vocab_size: 30000
```

#### Model Architecture
```yaml
model:
  d_model: 256
  n_heads: 8
  n_layers_enc: 3
  n_layers_dec: 3
  d_ff: 1024
  dropout: 0.0
```

#### Training
```yaml
experiment:
  epochs: 3
  lr: 0.0005
  lr_asc: 0.00025      # ASC 파라미터 별도 LR
  warmup_steps: 800
  clip_grad: 0.8
  seeds: [42]          # 재현성
```

### 2.3 ASCender 설정 (메인)

#### Component Weights
```yaml
asc_cfg:
  use_alignment: true
  use_cohesion: true
  use_separation: false

  w_align: 5.0         # 의미 기반
  w_coh: 10.0          # 위치 기반 (주력)
  w_sep: 0.0           # OFF
```

#### Stabilization
```yaml
  # 범위 제한
  clamp_min: -2.0
  clamp_max: 2.0
  gate_floor: 0.20
  gate_ceiling: 0.70

  # 자동 보정
  use_auto_calibrate: true
  target_ratio: 0.30

  # ALiBi 혼합
  use_alibi_mix: true
  alpha_start: 0.30     # 초반 ALiBi 70%
  alpha_end: 0.70       # 후반 ASC 70%
  alpha_schedule: "cosine"
```

#### Safety Limits
```yaml
  hard_max_ratio: 0.85
  gamma_min: 0.30
  gamma_cap: 4.0
```

### 2.4 비교 실험 설계

#### Ablation Study
```bash
# 1. Baseline
python src/train.py --config configs/baseline.yaml

# 2. ASCender (메인)
python src/train.py --config configs/ascender256.yaml

# 3. Safe vs Aggressive
python src/train.py --config configs/ascender_safe.yaml
python src/train.py --config configs/ascender_very_aggressive.yaml

# 4. Residual Path 효과
python src/train.py --config configs/ascender256_residual.yaml
```

#### Component Ablation
각 요소 제거 후 성능 측정:
1. Auto-calibration OFF
2. ALiBi Mix OFF
3. Gate Ceiling 제거
4. Per-head params OFF

---

## 📊 3. 결과 수집 체계

### 3.1 로그 구조
```
logs/
├── checkpoints/           # 모델 체크포인트
│   └── epoch_*.pt
├── heatmaps/              # 바이어스 시각화
│   └── bias_epoch_*.png
├── alpha/                 # α 스케줄 (ALiBi/Residual)
├── attn/                  # Attention 패턴
├── results_summary.csv    # 전체 실험 요약
└── additive_logs/
    └── metrics.pt         # PyTorch 텐서 메트릭
```

### 3.2 주요 메트릭

#### 성능 지표 (results_summary.csv)
```csv
experiment,seed,epoch,loss,ppl,acc,train_time
baseline,42,3,4.523,92.1,0.423,180s
ascender256,42,3,4.287,72.5,0.445,195s
```

#### Bias 진단 (로그 출력)
```
[L0] Bias Ratio: 0.287 (target: 0.30)
[L0] γ_eff=1.234, σ_eff=0.567
[L0] α_alibi=0.650 (step 1200/1000)
```

### 3.3 S3 동기화
```bash
# GPU 워커에서 자동 실행 (userdata)
aws s3 sync /home/ubuntu/logs s3://${BUCKET}/logs/ --exclude "*.pt"
aws s3 sync /home/ubuntu/checkpoints s3://${BUCKET}/checkpoints/
```

---

## 🚀 4. 실행 가이드

### 4.1 로컬 개발
```bash
# 환경 설정
conda activate ascender311

# 빠른 테스트 (1 epoch)
python src/train.py \
  --config configs/ascender256.yaml \
  --epochs 1 \
  --batch_size 2

# 전체 실험
python src/train.py --config configs/ascender256.yaml
```

### 4.2 GPU 워커 (EC2)
```bash
# SSH 접속
ssh -i ~/.ssh/your-key.pem ubuntu@<gpu-worker-ip>

# 실험 실행 (tmux 세션)
tmux new -s exp1
conda activate pytorch
cd /opt/ml/ascender
python src/train.py --config configs/ascender256.yaml

# 진행 상황 모니터링
tail -f logs/train.log
nvidia-smi -l 1  # GPU 사용률
```

### 4.3 결과 확인
```bash
# 로컬에서 S3 다운로드
aws s3 sync s3://ascender-experiments/logs/ ./logs/
aws s3 sync s3://ascender-experiments/checkpoints/ ./checkpoints/

# 분석
python src/analyze_results_general.py
```

---

## 🔧 5. 주요 스크립트

### 5.1 학습
```bash
src/train.py              # 메인 학습 루프
```

### 5.2 분석
```bash
src/analyze_results.py            # 단일 실험 분석
src/analyze_results_general.py    # 전체 실험 비교
src/compare_modes.py              # Mode별 비교 (additive/multiplicative)
src/compare_baselines.py          # Baseline vs ASCender
```

### 5.3 시각화
```bash
visualize_current_bias.py         # 현재 bias 패턴
diagnose_bias.py                  # Bias 진단 (여러 T 크기)
src/utils/visualize_attention_entropy.py  # Attention 엔트로피
```

### 5.4 유틸리티
```bash
extract_alpha.py                  # α 값 추출 (ALiBi/Residual)
scripts/measure_bias_effect.py    # Bias 영향도 측정
```

---

## 📋 6. 체크리스트

### 실험 시작 전
- [ ] Config 파일 검증 (`yaml.safe_load()`)
- [ ] Seed 고정 확인 (`seeds: [42]`)
- [ ] GPU 메모리 충분 (`nvidia-smi`)
- [ ] S3 버킷 접근 권한 (`aws s3 ls`)
- [ ] Git commit (코드 버전 기록)

### 실험 중
- [ ] Loss 발산 모니터링 (첫 100 step)
- [ ] Bias Ratio 추적 (0.2~0.4 범위)
- [ ] γ/σ 값 확인 (로그)
- [ ] GPU 사용률 >80% (효율성)

### 실험 종료 후
- [ ] 결과 S3 업로드 확인
- [ ] CSV 요약 생성 (`results_summary.csv`)
- [ ] 체크포인트 저장 (best epoch)
- [ ] 실험 노트 업데이트
- [ ] Config 파일 아카이브

---

## 🎯 7. 빠른 참조

### 주요 파라미터 범위
| 파라미터 | Conservative | Moderate | Aggressive |
|----------|--------------|----------|------------|
| `std_match_ratio` | 0.20 | 0.25 | 0.30~0.35 |
| `gate_ceiling` | 0.60 | 0.65 | 0.70 |
| `w_coh` | 5.0 | 7.5 | 10.0+ |
| `w_align` | 2.0 | 3.5 | 5.0+ |
| `gamma_cap` | 3.0 | 4.0 | 6.0 |

### 문제 해결
| 증상 | 원인 | 해결 |
|------|------|------|
| Loss 폭발 | Bias 과도 | `hard_max_ratio` 낮추기 |
| 효과 미미 | Bias 약함 | `gate_floor` 올리기, `std_match_ratio` 올리기 |
| 학습 불안정 | Auto-calib 간섭 | `use_auto_calibrate: false` |
| OOM | Batch/seq 크기 | `batch_size` 줄이기 |

### 연락처/리소스
- **코드**: `/Users/aepeul/ASCender/basic-transformer/`
- **인프라**: `/Users/aepeul/ASCender/terraform/`
- **문서**: `STABILITY_ELEMENTS_REPORT.md` (상세)
- **Config**: `configs/ascender256.yaml` (메인)

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-11-06
