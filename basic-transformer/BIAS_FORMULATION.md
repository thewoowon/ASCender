# ASCender Bias 수식 정의

## 전체 Bias 구조

ASCender는 attention logit에 다음과 같은 additive bias를 적용한다:

```
Attention(Q, K, V) = softmax((QK^T / √d_k) + B) V
```

여기서 bias term **B**는 세 가지 component의 가중합으로 구성된다:

```
B_{raw} = w_A · B_A + w_S · B_S + w_C · B_C
B = γ · σ · clamp(B_{raw})
```

- **γ** (gamma): Learnable scale parameter (per-head 또는 global)
- **σ** (gate): Learnable gate ∈ [σ_floor, σ_ceiling] (optional)
- **clamp**: [-c_min, c_max] 범위로 clipping

---

## 1. Alignment Bias (B_A)

**목적**: 의미적으로 유사한 토큰들이 서로 attend하도록 유도

### 수식

```
B_A^{(i,j)} = sim(q_i, k_j) / τ
```

여기서:
- **sim(q, k)**: Cosine similarity 또는 normalized dot product
- **τ**: Temperature parameter (default: 1.0)

### 구현 방식 (2가지)

#### (a) QK-based alignment (기본)
```
q̂_i = q_i / ||q_i||,  k̂_j = k_j / ||k_j||

B_A^{(i,j)} = (q̂_i · k̂_j) / τ
           = (q_i · k_j) / (||q_i|| · ||k_j|| · τ)
```

**특징**:
- Query/Key 벡터의 방향 유사도 측정
- 이미 attention 계산에서 사용하는 QK를 재활용
- Scale-invariant (벡터 크기 영향 제거)

#### (b) Pre-projection based alignment (선택)
```
B_A^{(i,j)} = (ĥ_i^q · ĥ_j^k) / τ
```

여기서 h^q, h^k는 Q/K projection 이전의 원본 embedding

**특징**:
- 원본 embedding 공간의 유사도 사용
- Projection layer의 영향 배제
- 더 "raw"한 의미적 유사도 측정

### 코드 구현
```python
# ascender_bias.py:379-391
if self.cfg.use_alignment and self.cfg.w_align != 0.0:
    if self.cfg.align_source == "qk":
        qn = F.normalize(qh, dim=-1)  # L2 normalization
        kn = F.normalize(kh, dim=-1)
        align = torch.matmul(qn, kn.transpose(-2, -1))  # (B,H,T,S)
    else:  # preproj
        qn = F.normalize(pre_q, dim=-1)
        kn = F.normalize(pre_k, dim=-1)
        align = torch.matmul(qn, kn.transpose(-2, -1))

    if self.cfg.temperature != 1.0:
        align = align / self.cfg.temperature

    bias = bias + self.cfg.w_align * align
```

### 해석
- **양수 bias**: 유사한 토큰에 더 높은 attention 확률
- **Temperature τ**:
  - τ > 1: 유사도 차이 완화 (smooth)
  - τ < 1: 유사도 차이 강조 (sharp)

---

## 2. Separation Bias (B_S)

**목적**: 가까운 토큰들 간 반발력 추가 (경계 형성, 다양성 유도)

### 수식

```
B_S^{(i,j)} = -exp(-Δ_{ij}^2 / (2σ_S^2)) · m_{band}^{(i,j)} · m_{dir}^{(i,j)}
```

여기서:
- **Δ_{ij} = |i - j|**: 토큰 간 절대 거리
- **σ_S**: Separation kernel width (default: 2.0)
- **m_band**: Band-pass mask (거리 제한)
- **m_dir**: Directionality mask (past_only인 경우 i ≥ j만 허용)

### 상세 정의

#### Gaussian Kernel
```
G_S(Δ) = exp(-Δ^2 / (2σ_S^2))
```

**특성**:
- Δ = 0 (자기자신): G_S = 1 (최대 반발)
- Δ = σ_S: G_S ≈ 0.61
- Δ = 2σ_S: G_S ≈ 0.14
- Δ >> σ_S: G_S → 0 (반발 없음)

**Effective Range**: 약 ±2σ_S (예: σ_S=2.0 → 약 ±4 tokens)

#### Band-pass Mask
```
m_band^{(i,j)} = 1  if Δ_min ≤ |i-j| ≤ Δ_max
                 0  otherwise
```

**목적**: 너무 먼 토큰에는 bias 적용 안 함 (효율성, 안정성)

#### Directionality Mask
```
m_dir^{(i,j)} = 1  if i ≥ j  (causal, past_only=true)
                1  if True    (bidirectional, past_only=false)
```

### 코드 구현
```python
# ascender_bias.py:399-401
if self.cfg.use_separation and self.cfg.w_sep != 0.0:
    sep = self._gauss(rel_abs, self.cfg.sigma_sep) * band * dirmask
    bias = bias - self.cfg.w_sep * sep.view(1, 1, T, S)

# Gaussian helper (line 249-251)
@staticmethod
def _gauss(rel_abs: torch.Tensor, sigma: float) -> torch.Tensor:
    σ = max(1e-6, float(sigma))
    return torch.exp(-(rel_abs ** 2) / (2.0 * σ * σ))
```

### 해석
- **음수 bias** (반발): 가까운 토큰에 낮은 attention 확률
- **효과**:
  - Local diversity (같은 위치만 계속 보지 않음)
  - Boundary detection (문장/구 경계에서 attention 단절)
  - Over-smoothing 방지

**예시** (σ_S=2.0):
```
i=5일 때:
j=5 (자신):   B_S = -w_S × 1.00  (최대 반발)
j=6,4 (±1):   B_S = -w_S × 0.88
j=7,3 (±2):   B_S = -w_S × 0.61
j=9,1 (±4):   B_S = -w_S × 0.14
j=13,-3 (±8): B_S ≈ -w_S × 0.02  (거의 없음)
```

---

## 3. Cohesion Bias (B_C)

**목적**: 대각선 중심으로 local neighborhood 강화 (근접 토큰 선호)

### 수식

```
B_C^{(i,j)} = +exp(-Δ_{ij}^2 / (2σ_C^2)) · m_{band}^{(i,j)} · m_{dir}^{(i,j)}
```

여기서:
- **σ_C**: Cohesion kernel width (default: 50.0)
- 나머지는 Separation과 동일한 mask 적용

### 상세 정의

#### Gaussian Kernel
```
G_C(Δ) = exp(-Δ^2 / (2σ_C^2))
```

**특성**:
- Δ = 0: G_C = 1 (최대 응집)
- Δ = σ_C: G_C ≈ 0.61
- Δ = 2σ_C: G_C ≈ 0.14
- Δ >> σ_C: G_C → 0 (응집 없음)

**Effective Range**: 약 ±2σ_C (예: σ_C=50.0 → 약 ±100 tokens)

### 코드 구현
```python
# ascender_bias.py:403-405
if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
    coh = self._gauss(rel_abs, self.cfg.sigma_coh) * band * dirmask
    bias = bias + self.cfg.w_coh * coh.view(1, 1, T, S)
```

### 해석
- **양수 bias** (인력): 가까운 토큰에 높은 attention 확률
- **효과**:
  - Local context 선호
  - Sliding window attention 유사 효과
  - 대각선 중심 밴드 구조 형성

**예시** (σ_C=50.0, seq_len=256):
```
i=128일 때:
j=128 (자신):  B_C = +w_C × 1.00  (최대 응집)
j=178,78 (±50):  B_C = +w_C × 0.61
j=228,28 (±100): B_C = +w_C × 0.14
j=256,0 (±128):  B_C = +w_C × 0.07
```

---

## 4. Separation vs Cohesion 비교

| 특성 | Separation (B_S) | Cohesion (B_C) |
|------|------------------|----------------|
| **부호** | 음수 (반발) | 양수 (인력) |
| **목적** | 다양성, 경계 형성 | 근접성, 응집 |
| **σ 범위** | 좁음 (σ_S ≈ 2.0) | 넓음 (σ_C ≈ 50.0) |
| **효과 범위** | ±4 tokens | ±100 tokens |
| **생물학적 비유** | Boids repulsion | Boids cohesion |
| **Vision 비유** | Edge detection | Blur/Smoothing |

### 상호작용
```
Total Position Bias = B_S + B_C
                    = w_C·G_C(Δ) - w_S·G_S(Δ)
```

**예시** (w_S=1.5, w_C=5.0, σ_S=2.0, σ_C=50.0):
```
Δ=0:  B_pos = 5.0×1.0 - 1.5×1.0 = +3.5 (net 응집)
Δ=2:  B_pos = 5.0×0.99 - 1.5×0.61 = +4.04 (강한 응집)
Δ=4:  B_pos = 5.0×0.97 - 1.5×0.14 = +4.64 (매우 강한 응집)
Δ=10: B_pos = 5.0×0.92 - 1.5×0.0 ≈ +4.60 (응집 지배)
```

**해석**:
- **σ_C >> σ_S**이므로 cohesion이 wide range에 걸쳐 작용
- Separation은 매우 가까운 거리(±4 tokens)에서만 효과
- 전체적으로 **local cohesion 선호** 패턴 형성

---

## 5. 최종 Bias 계산 파이프라인

```
(1) Raw Bias 계산:
    B_raw = w_A·B_A + w_S·B_S + w_C·B_C

(2) Optional: Row-wise centering (enable_centering=true)
    B_centered = B_raw - mean(B_raw, dim=j)

    ⚠️ 주의: Centering은 global structure를 파괴함!
    언어 모델링에서는 false 권장

(3) Clamp (안정성):
    B_clamped = clamp(B_centered, -c_min, c_max)

    기본값: [-12.0, 12.0]

(4) Optional: ALiBi convex mix (use_alibi_mix=true)
    B_mixed = α·B_clamped + (1-α)·B_ALiBi

    여기서 α ∈ [0,1]는 schedule 가능

(5) Learnable scaling & gating:
    γ = exp(γ_log)  ∈ [γ_min, γ_cap]
    σ = sigmoid(σ_logit) · (σ_ceiling - σ_floor) + σ_floor

    B_final = γ · σ · B_mixed

(6) Optional: Auto-calibration (training 중)
    γ ← γ · adjust_factor

    adjust_factor 기반: std(B) / std(QK^T)를 target_ratio로 맞춤
```

### 코드 흐름
```python
# ascender_bias.py:374-433
def forward(self, qh, kh, pre_q=None, pre_k=None, scores_std=None):
    # (1) Raw bias
    bias = torch.zeros((B, H, T, S), device=device)

    if use_alignment:
        align = normalized_dot_product(qh, kh) / temperature
        bias += w_align * align

    if use_separation:
        sep = gaussian(|i-j|, sigma_sep) * masks
        bias -= w_sep * sep

    if use_cohesion:
        coh = gaussian(|i-j|, sigma_coh) * masks
        bias += w_coh * coh

    # (2) Centering (optional, default=false)
    if enable_centering:
        bias = bias - bias.mean(dim=-1, keepdim=True)

    # (3) Clamp
    bias = clamp(bias, clamp_min, clamp_max)

    # (4) ALiBi mix (optional)
    if use_alibi_mix:
        bias = alpha * bias + (1-alpha) * alibi_bias

    # (5) Scale & Gate
    gamma_eff = exp(gamma_log).clamp(gamma_min, gamma_cap)
    scaled = gamma_eff * bias

    if use_gate:
        gate_eff = sigmoid(gate_param) * (ceiling - floor) + floor
        scaled = scaled * gate_eff

    # (6) Auto-calibration (training only)
    if use_auto_calibrate and training:
        ratio_current = std(scaled) / std(scores)
        adjust = target_ratio / ratio_current
        gamma_log += log(adjust) * momentum

    return scaled
```

---

## 6. 하이퍼파라미터 요약

### Component Weights
```
w_A ∈ ℝ     (default: 3.5)   Alignment weight
w_S ∈ ℝ     (default: 1.5)   Separation weight (repulsion)
w_C ∈ ℝ     (default: 5.0)   Cohesion weight (attraction)
```

### Kernel Widths
```
σ_S ∈ ℝ₊    (default: 2.0)   Separation range (narrow, ~4 tokens)
σ_C ∈ ℝ₊    (default: 50.0)  Cohesion range (wide, ~100 tokens)
```

### Scale & Gate
```
γ ∈ [γ_min, γ_cap]  (default: [0.5, 8.0])   Learnable scale
σ ∈ [σ_floor, σ_ceiling]  (default: [0.1, 0.9])  Learnable gate
```

### Stability
```
c_min, c_max  (default: -12.0, 12.0)  Clamp range
τ             (default: 1.0)          Alignment temperature
```

### Masks
```
Δ_min, Δ_max  (default: 0, 128)      Band-pass range
past_only     (default: true)        Causal masking
```

---

## 7. 수식 정리 (Compact Form)

최종 bias는 다음과 같이 요약할 수 있다:

```
B(i,j) = γ · σ · clamp[
    w_A · (q̂ᵢ·k̂ⱼ)/τ
    + w_C · exp(-Δᵢⱼ²/2σ_C²)
    - w_S · exp(-Δᵢⱼ²/2σ_S²)
] · m_band(Δᵢⱼ) · m_dir(i,j)

여기서:
  Δᵢⱼ = |i - j|
  q̂ᵢ = qᵢ/||qᵢ||,  k̂ⱼ = kⱼ/||kⱼ||
  γ = exp(γ_log) ∈ [γ_min, γ_cap]
  σ = σ_floor + (σ_ceiling - σ_floor)·sigmoid(σ_logit)
```

**Attention 계산**:
```
Attention(Q,K,V) = softmax((QK^T/√dₖ + B) ⊙ M_causal) V
```

---

## 8. 실험에서 사용된 설정 (WikiText-103)

```yaml
use_alignment:  false
use_separation: true
use_cohesion:   true

w_align: 3.5
w_sep:   1.5
w_coh:   5.0

sigma_sep: 2.0
sigma_coh: 50.0

temperature: 1.0

clamp_min: -12.0
clamp_max:  12.0

enable_centering: false

past_only: true
band_min: 0
band_max: 128

per_head_scale: true
global_scale_init: 1.0

use_gate: true
gate_init: 0.0
gate_floor: 0.10
gate_ceiling: 0.90
```

**결과 패턴**:
- Diagonal mean: 1.93
- Off-diagonal mean: 0.56
- Bias std: 0.91
- 대각선 중심의 cohesion 밴드 구조 명확히 형성

---

## 9. 주요 디자인 결정 및 근거

### (1) Why Gaussian kernels?
- **부드러운 감쇠**: Hard cutoff 대신 smooth transition
- **미분 가능**: Backpropagation 지원
- **해석 가능**: σ가 effective range 결정

### (2) Why σ_C >> σ_S?
- **다른 스케일**: Separation은 local (±4), Cohesion은 wide (±100)
- **생물학적 영감**: Boids에서도 separation < cohesion range
- **실용적 효과**: Local diversity + global cohesion

### (3) Why learnable γ and σ?
- **Task adaptivity**: 모델이 bias 강도를 학습으로 조절
- **Weak prior**: 너무 강한 inductive bias 방지
- **Gate로 무시 가능**: 불필요하면 σ→0으로 학습 가능

### (4) Why no centering?
- **Global structure 보존**: 언어에는 절대 위치도 중요
- **실험 결과**: Centering 시 성능 저하 관찰
- **Diagnostic value**: 원본 bias 분포 유지로 분석 용이

---

## 10. 한계 및 향후 개선 방향

### 현재 한계
1. **Fixed σ**: Kernel width가 고정 → Adaptive σ 필요
2. **Position-only**: Separation/Cohesion이 거리만 고려 → Content-dependent 필요
3. **Full attention**: Bias 추가해도 O(n²) 유지 → Sparse와 결합 필요

### 제안 개선
1. **Learnable σ**:
   ```
   σ_C = exp(σ_C_log) ∈ [σ_min, σ_max]
   ```
2. **Content-dependent kernels**:
   ```
   σ_C(i) = f(hᵢ)  (MLP or attention-based)
   ```
3. **Bias-guided sparsity**:
   ```
   Select top-k positions based on B(i,j) before softmax
   ```
