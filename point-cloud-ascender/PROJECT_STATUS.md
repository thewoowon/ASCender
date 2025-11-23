# 🚀 Point Cloud ASCender - 프로젝트 현재 상태

**생성일**: 2025-11-11
**목표**: Dynamic Point Cloud Transformers with Boids-Inspired Bias

---

## ✅ 완료된 작업 (Phase 1)

### 1. 기존 연구 조사 ✓
- **Point Transformer (2021)** 논문 분석 완료
  - 핵심 메커니즘: `δ = θ(p_i - p_j)` (MLP 기반, Black box)
  - Vector attention 사용
  - k=16 neighbors, local attention

- **데이터셋 조사** 완료
  - MSRAction3D: 567 sequences, 20 actions (추천)
  - NTU RGB+D 120: 114,480 samples, 120 actions
  - KITTI Tracking: LiDAR sequences

- **GitHub 구현** 찾음
  - POSTECH-CVLab/point-transformer (official)
  - Pointcept/Pointcept (comprehensive)

### 2. 차별점 분석 ✓
문서: `DIFFERENTIATION_ANALYSIS.md`

**핵심 차별점 6가지:**
1. **해석 가능성**: MLP (Black box) → Boids 3-component (명시적)
2. **컴포넌트 분해**: 불가능 → A/S/C 개별 ablation 가능
3. **시간 확장**: Static → Dynamic Boids (원래 강점)
4. **혼합 전략**: 고정 → RBP (α로 자동 조절)
5. **멀티스케일**: k=16 고정 → Per-head adaptive σ
6. **이론적 근거**: 경험적 → 물리 법칙 (Boids)

**결론: 기우가 아니다! 명확한 차별점 존재.**

### 3. 핵심 코드 구현 ✓
파일: `src/models/point_ascender_bias.py`

**구현된 기능:**
```python
class PointAscenderBias(nn.Module):
    """
    Inputs:
      - qh, kh: Query/Key heads (B, H, T, d)
      - xyz_q, xyz_k: 3D coordinates (B, T, 3)
      - normals_q, normals_k: Surface normals (B, T, 3)

    Output:
      - bias: (B, H, Tq, Tk) additive bias
    """

    def compute_alignment():
        # Normal similarity: cos(θ)
        # OR feature similarity: q·k

    def compute_separation():
        # -exp(-||Δxyz||²/σ_sep²)
        # Far points → large negative bias

    def compute_cohesion():
        # exp(-||Δxyz||²/σ_coh²)
        # Close points → large positive bias
```

**특징:**
- ✅ 3D Euclidean distance 사용
- ✅ Normal-based alignment
- ✅ Per-head adaptive σ, γ
- ✅ Learnable gate
- ✅ Temporal smoothing (for dynamic clouds)
- ✅ Residual Bias Path 준비

---

## 📋 다음 단계 (Phase 2)

### Step 1: 환경 설정 (30분)
```bash
cd point-cloud-ascender
pip install -r requirements.txt
# 또는 conda environment
```

### Step 2: Point Transformer baseline fork (2-3시간)
```bash
git clone https://github.com/POSTECH-CVLab/point-transformer.git
# ASCender bias module 통합
```

### Step 3: MSRAction3D 데이터 로더 (1-2일)
- [ ] 데이터 다운로드
- [ ] Depth → Point cloud 변환
- [ ] Normal estimation
- [ ] DataLoader 작성

### Step 4: Quick PoC 실험 (2-3일)
**실험 설계:**
```yaml
Experiment 1: Baseline
  model: PointTransformer
  dataset: MSRAction3D
  epochs: 50

Experiment 2: ASCender (fixed α=0.5)
  model: PointTransformer + ASCender
  use_residual_path: False
  epochs: 50

Experiment 3: ASCender RBP (learnable α)
  model: PointTransformer + ASCender
  use_residual_path: True
  epochs: 50
```

**측정 지표:**
- Accuracy
- α values (per layer, per head)
- Component ablation (A only, S only, C only, A+S+C)

### Step 5: 결과 분석 및 판단 (1일)
**성공 조건:**
1. α ≠ 0.5 (바이어스가 실제로 기여)
2. Accuracy ≥ baseline (최소 동등)
3. 컴포넌트별 차별적 기여 관찰

**실패 시:**
- 바이어스 강도 증가
- Static ModelNet40으로 pivot
- 또는 다른 modality 고려

---

## 📁 프로젝트 구조

```
point-cloud-ascender/
├── README.md                           # 프로젝트 개요
├── DIFFERENTIATION_ANALYSIS.md         # 차별점 분석
├── PROJECT_STATUS.md                   # 현재 상태 (이 파일)
├── point_transformer_paper.pdf         # 참고 논문
├── requirements.txt                    # 의존성
├── src/
│   ├── models/
│   │   ├── point_ascender_bias.py     # ✅ 구현 완료
│   │   ├── point_transformer.py       # TODO: Baseline
│   │   └── point_multihead_attn.py    # TODO: MHA + ASCender
│   ├── data/
│   │   ├── msraction3d_loader.py      # TODO
│   │   └── point_cloud_utils.py       # TODO
│   └── utils/
│       ├── spatial_kernels.py         # TODO
│       └── normal_estimation.py       # TODO
├── configs/
│   ├── baseline_msraction3d.yaml      # TODO
│   └── ascender_msraction3d.yaml      # TODO
├── experiments/
│   └── quick_poc.py                   # TODO
└── tests/
    └── test_spatial_bias.py           # TODO
```

---

## 🎯 핵심 가설 (실험으로 검증)

### Hypothesis 1: α가 0.5에서 벗어날 것
- **WikiText (1D)**: α ≈ 0.5 (바이어스 약함)
- **Point Cloud (3D)**: α ≠ 0.5 (진짜 물리 공간)
- **측정**: `logs/alpha/alpha_epoch*.json` 분석

### Hypothesis 2: 컴포넌트별 차별적 기여
- **Alignment**: Shape matching에서 중요
- **Separation**: Noisy data에서 중요
- **Cohesion**: Tracking/Temporal에서 중요
- **측정**: Ablation study (A, S, C, A+S, A+C, S+C, A+S+C)

### Hypothesis 3: Dynamic에서 더 강함
- **Static (ModelNet)**: ASCender ≈ Point Transformer
- **Dynamic (MSRAction3D)**: ASCender > Point Transformer
- **측정**: Accuracy on action recognition

---

## 💡 예상 결과 시나리오

### 시나리오 A: 대성공 🎉
- α ≈ 0.3 (바이어스 강하게 기여)
- Accuracy > baseline (+2~5%)
- 각 component 명확한 역할
→ **Full research project 진행**

### 시나리오 B: 부분 성공 ✓
- α ≈ 0.4~0.6 (중간 기여)
- Accuracy ≈ baseline (±1%)
- 해석가능성 강점
→ **Interpretability 강조 논문**

### 시나리오 C: 실패 ⚠️
- α ≈ 0.5 (바이어스 약함)
- Accuracy < baseline
→ **바이어스 강도 증가 OR pivot**

---

## 📊 타임라인

### Week 1 (현재)
- [x] 연구 조사
- [x] 차별점 분석
- [x] 핵심 코드 구현
- [ ] 환경 설정

### Week 2
- [ ] Baseline fork
- [ ] Data loader
- [ ] Quick PoC 실험

### Week 3
- [ ] 결과 분석
- [ ] 방향 결정
- [ ] (성공 시) Full project 계획

---

## 🔗 참고 자료

### 논문
- Point Transformer (Zhao et al., ICCV 2021)
- Boids (Reynolds, SIGGRAPH 1987)
- ASCender NLP baseline (우리 프로젝트)

### 코드
- https://github.com/POSTECH-CVLab/point-transformer
- https://github.com/Pointcept/Pointcept

### 데이터셋
- MSRAction3D: http://research.microsoft.com/en-us/um/people/zliu/ActionRecoRsrc/
- NTU RGB+D: https://rose1.ntu.edu.sg/dataset/actionRecognition/

---

## 📝 다음 작업

**즉시 (오늘):**
1. ✅ 환경 설정 (`pip install -r requirements.txt`)
2. Point Transformer baseline 코드 다운로드
3. MSRAction3D 데이터셋 다운로드

**이번 주:**
1. Data loader 작성
2. Baseline 학습
3. ASCender 통합

**중요:** Quick PoC 결과에 따라 full project 진행 여부 결정!

---

**Status**: 🟢 Phase 1 완료, Phase 2 준비 중
**Next Milestone**: Quick PoC 실험 (2-3일 소요 예상)
