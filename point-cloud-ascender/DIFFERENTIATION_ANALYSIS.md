# Point Transformer vs ASCender: 명확한 차별점 분석

## 📊 Point Transformer의 핵심 메커니즘 (2021)

### **수식 분석**

**Vector Attention (Eq. 3):**
```
y_i = Σ_{x_j ∈ X(i)} ρ(γ(φ(x_i) - ψ(x_j) + δ)) ⊙ (α(x_j) + δ)
```

**Position Encoding (Eq. 4):**
```
δ = θ(p_i - p_j)
```
- `p_i, p_j`: 3D 좌표 (x, y, z)
- `θ`: **MLP (2 linear layers + ReLU)** - **학습됨, 해석 불가능**

### **작동 방식**
1. **상대 위치**: `p_i - p_j` (3D 벡터)
2. **MLP 변환**: `δ = MLP(p_i - p_j)` → 고차원 임베딩
3. **Attention 생성**: `γ(feature_diff + δ)`
4. **Feature 변환**: `α(x_j) + δ`

### **특징**
- ✅ 3D 좌표 직접 사용
- ✅ 학습 가능한 위치 인코딩
- ❌ **Black box** - δ가 무엇을 학습했는지 알 수 없음
- ❌ 컴포넌트 분해 불가능
- ❌ Static point clouds에 주로 사용

---

## 🆚 ASCender의 차별점

### **1. 명시적 3-Component Decomposition**

**Point Transformer:**
```python
δ = MLP(p_i - p_j)  # Black box, 해석 불가
```

**ASCender:**
```python
# 명시적 Boids 3-component
δ = w_align * Alignment(p_i, p_j, normals) +
    w_sep   * Separation(||p_i - p_j||) +
    w_coh   * Cohesion(||p_i - p_j||)
```

**각 컴포넌트의 물리적 의미:**
- **Alignment**: 표면 법선 일관성 (같은 방향 = 같은 object)
- **Separation**: 노이즈/이상치 억제 (너무 멀면 다른 object)
- **Cohesion**: 국소 구조 유지 (가까우면 같은 part)

### **2. Ablation으로 기여도 측정 가능**

**Point Transformer:**
- "MLP 없으면 성능 떨어짐" ← 당연하지만, **왜?** 알 수 없음

**ASCender:**
```python
# 각 컴포넌트 제거하면서 실험
A only:   accuracy = X%  → Alignment의 기여
S only:   accuracy = Y%  → Separation의 기여
C only:   accuracy = Z%  → Cohesion의 기여
A+S+C:    accuracy = W%  → 시너지 효과
```
→ **정확히 어떤 컴포넌트가 얼마나 기여하는지 측정 가능**

### **3. Dynamic Point Clouds 확장**

**Point Transformer:**
```python
# 각 프레임 독립 처리
for frame_t in sequence:
    output_t = PointTransformer(frame_t)
```

**ASCender:**
```python
# 시간적 일관성을 Boids 동역학으로 모델링
Cohesion_t = exp(-||p_i(t) - p_j(t)||²/σ²) * motion_consistency(t-1, t)
# Cohesion이 물체 persistence 보장
# Alignment가 motion coherence 추적
```

**차이:**
- Point Transformer: 공간만 고려 (static bias)
- ASCender: **공간 + 시간 동역학** (temporal Boids)

### **4. Residual Bias Path (RBP)**

**Point Transformer:**
```python
# 전부 학습 OR 전부 inductive bias
attention = softmax(Q·K^T + δ)  # δ는 항상 적용
```

**ASCender:**
```python
# 자동으로 혼합 비율 학습
attention = α * learned_attn + (1-α) * boids_bias
```

**효과:**
- α ≈ 1: 학습된 attention 선호 (복잡한 패턴)
- α ≈ 0: 구조적 bias 선호 (물리적 법칙)
- **모델이 자동으로 최적 혼합 학습**

### **5. Per-head Adaptive Scales**

**Point Transformer:**
```python
# 모든 헤드가 같은 k=16 neighbors 사용
X(i) = kNN(i, k=16)  # 고정
```

**ASCender:**
```python
# 각 헤드가 다른 σ 학습 가능
Head 0: σ_coh = 0.1m  (local details)
Head 4: σ_coh = 0.5m  (mid-level structure)
Head 7: σ_coh = 2.0m  (global shape)
```

**결과:**
- Point Transformer: 단일 스케일
- ASCender: **자동 멀티스케일 출현**

---

## 🎯 핵심 차별점 요약

| 측면 | Point Transformer | ASCender |
|---|---|---|
| **위치 인코딩** | MLP(p_i - p_j) - Black box | Boids 3-component - 명시적 |
| **해석 가능성** | ❌ 불가능 | ✅ 각 컴포넌트 기여도 측정 |
| **Ablation** | "MLP 필요함" (당연) | A/S/C 개별 효과 정량화 |
| **시간 확장** | Static (프레임 독립) | Dynamic (시간 동역학) |
| **혼합 전략** | 고정 (항상 δ 사용) | RBP (α로 자동 조절) |
| **멀티스케일** | 고정 k=16 | Per-head adaptive σ |
| **이론적 근거** | 경험적 ("잘 됨") | 물리 법칙 (Boids) |

---

## 💡 연구 기여도

### **1. Interpretability (가장 큰 강점)**

**질문:** "왜 이 점들이 같은 object로 분류됐나?"

**Point Transformer 답변:**
- "MLP가 그렇게 학습했습니다" ← 설명 끝

**ASCender 답변:**
- "Alignment score = 0.85 (법선 유사)"
- "Cohesion score = 0.92 (거리 가까움)"
- "Separation score = 0.05 (이상치 아님)"
- → **명확한 근거 제시 가능**

### **2. Domain Transfer**

**Point Transformer:**
- 새 도메인 → 전체 MLP 재학습 필요

**ASCender:**
- Boids 컴포넌트는 범용 물리 법칙
- 가중치 (w_align, w_sep, w_coh)만 조정
- **Few-shot learning에 유리**

### **3. Dynamic Point Clouds (최고 차별점)**

**기존 연구:** Static point clouds 위주 (ModelNet, S3DIS)

**ASCender:**
- **시간적 Boids 동역학** (원래 강점 활용)
- Action recognition (MSRAction3D, NTU RGB+D)
- Object tracking (KITTI)
- **이 영역은 2021년에 충분히 탐구 안 됨**

---

## 📌 "단순한 거리 기반 아닌가?" 답변

### ❌ **단순하지 않습니다:**

1. **Point Transformer의 MLP:**
   - 입력: `p_i - p_j` (3D 벡터)
   - 출력: 고차원 임베딩 (e.g., 256-dim)
   - **무엇을 학습?** → 모름

2. **ASCender의 Boids:**
   - 입력: `p_i - p_j` (3D 벡터) + normals
   - 출력:
     - Alignment: `cos(θ) between normals`
     - Separation: `-exp(-||Δp||²/σ_sep²)`
     - Cohesion: `exp(-||Δp||²/σ_coh²)`
   - **무엇을 계산?** → **명확한 물리적 관계**

3. **복잡도 비교:**
   - Point Transformer MLP: ~1024 parameters (per layer)
   - ASCender components: ~20 parameters (w, σ, γ)
   - **하지만 해석 가능성은 ASCender가 압도적**

---

## 🚀 실험으로 증명할 것

### **Hypothesis 1: α가 0.5에서 벗어날 것**
- **WikiText (1D):** α ≈ 0.5 (바이어스 약함)
- **Point Cloud (3D):** α ≠ 0.5 (진짜 물리 공간)

### **Hypothesis 2: 컴포넌트별 차별적 기여**
- **Noisy data:** Separation 중요 ↑
- **Tracking:** Cohesion 중요 ↑
- **Shape matching:** Alignment 중요 ↑

### **Hypothesis 3: Dynamic에서 더 강함**
- **Static (ModelNet):** ASCender ≈ Point Transformer
- **Dynamic (MSRAction3D):** ASCender > Point Transformer

---

## 📝 결론: 기우가 아닙니다!

### ✅ **명확한 차별점 존재:**
1. **해석 가능성** (Black box → 명시적 컴포넌트)
2. **시간 확장** (Static → Dynamic Boids)
3. **Adaptive 혼합** (고정 → RBP α 학습)

### ⚠️ **현실적 평가:**
- 단순 ModelNet classification에서는 큰 차이 없을 수도
- **하지만**: Dynamic, Few-shot, Interpretability 필요한 곳에서 빛남
- 논문 기여도: **충분히 novelty 있음**

### 🎯 **전략:**
- Quick PoC로 빠른 검증
- Dynamic Point Clouds에 집중 (최고 차별점)
- Interpretability 강조 (의료, 로보틱스 등에서 중요)

---

**다음 단계: Point Cloud용 ASCender Bias 모듈 구현**
