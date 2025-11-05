# ASCender 모델 안정화 요소 및 비교 지표 정리

> **작성일**: 2025-11-06
> **목적**: Boids 바이어스 설계 외 안정화 요소 및 부가 기능 체계화

---

## 📋 목차

1. [안정화 요소 (Stabilization Elements)](#1-안정화-요소)
2. [부가 요소 (Additional Features)](#2-부가-요소)
3. [아키텍처 확장 옵션](#3-아키텍처-확장-옵션)
4. [비교 지표 체계](#4-비교-지표-체계)
5. [코드 위치 참조](#5-코드-위치-참조)

---

## 1. 안정화 요소

### 1.1 Centering (중심화)
**위치**: `ascender_bias.py:407-410`

```python
if getattr(self.cfg, "enable_centering", False):
    bias = bias - bias.mean(dim=-1, keepdim=True)
```

- **목적**: 바이어스의 평균을 0으로 조정하여 softmax에 미치는 영향 정규화
- **기본값**: `False` (비활성화)
- **주의사항**: 전역 구조 정보 손실 가능성으로 디버깅용으로만 사용 권장
- **설정**: `asc_cfg.enable_centering`

### 1.2 Clamping (클램핑)
**위치**: `ascender_bias.py:39-40, 412`

```python
clamp_min: float = -10.0
clamp_max: float = 10.0
# ...
bias = bias.clamp_(float(self.cfg.clamp_min), float(self.cfg.clamp_max))
```

- **목적**: 바이어스 값의 범위를 제한하여 수치 불안정성 방지
- **기본값**: `[-10.0, 10.0]`
- **실제 사용**: 현재 config에서 `[-2.0, 2.0]`로 더 좁게 설정
- **적용 시점**: γ-scale 이전 (raw bias)
- **설정**: `asc_cfg.clamp_min`, `asc_cfg.clamp_max`

### 1.3 Auto-Calibration (자동 보정)
**위치**: `ascender_bias.py:61-67, 434-488`

```python
use_auto_calibrate: bool = False
target_ratio: float = 0.30
calibrate_step_clamp_lo: float = 0.90
calibrate_step_clamp_hi: float = 1.12
ema_momentum: float = 0.90
```

#### 동작 원리
1. **비율 측정**: `std(bias) / std(scores)` 계산
2. **EMA 추적**: 지수이동평균으로 비율 추적
3. **γ 자동 조정**: 타깃 비율에 수렴하도록 log-γ 업데이트
4. **게이트 비례 제어**: 비율 오차에 비례하여 σ 조정

#### 조정 로직
```python
# γ 조정 (10%로 감소한 gentle update)
adj_h = torch.clamp(target / ratio_h, min=lo, max=hi)
gentle_adj = (adj_h - 1.0) * 0.1 + 1.0
gamma_log.data.add_(gentle_adj.log())

# σ 게이트 조정 (k=0.02, 비례 제어)
err_h = (ratio_h - target)  # 양수: 너무 강함 → 닫기
delta = (-k * err_h)
gate_param.data.add_(delta)
```

- **목적**: 학습 중 bias/score 비율을 타깃값(0.30)으로 자동 유지
- **워밍업**: `calibrate_warmup_steps` 이후 활성화
- **설정**: `asc_cfg.use_auto_calibrate`, `asc_cfg.target_ratio`

### 1.4 Gate Ceiling/Floor (게이트 범위 제한)
**위치**: `ascender_bias.py:57-59, 340-347`

```python
gate_floor: float = 0.15      # 최소 게이트 값
gate_ceiling: float = 0.85    # 최대 게이트 값
# ...
g_raw = torch.sigmoid(gate_param)
g = gate_floor + (1.0 - gate_floor) * g_raw
g = torch.minimum(g, torch.as_tensor(float(gate_ceiling), device=g.device))
```

#### 게이트 매핑 과정
1. **Logit → Sigmoid**: `gate_param` → `[0, 1]`
2. **Floor 적용**: `[0, 1]` → `[floor, 1]`
3. **Ceiling 적용**: `[floor, 1]` → `[floor, ceiling]`

- **목적**:
  - Floor: 바이어스가 완전히 꺼지지 않도록 최소값 보장
  - Ceiling: 바이어스가 score를 압도하지 않도록 상한 제한
- **기본값**: `[0.15, 0.85]`
- **실제 사용**: `[0.20, 0.70]` (더 보수적)
- **설정**: `asc_cfg.gate_floor`, `asc_cfg.gate_ceiling`

### 1.5 ALiBi Convex Mix (ALiBi 혼합)
**위치**: `ascender_bias.py:80-86, 414-420`

```python
use_alibi_mix: bool = True
alpha_start: float = 0.5      # 초기 ASC 비중
alpha_end: float = 0.6        # 최종 ASC 비중
alpha_schedule: Literal["none", "cosine"] = "none"
alpha_total_steps: int = 0
```

#### 스케줄링
```python
# Cosine interpolation
t = min(max(step / total, 0.0), 1.0)
alpha = a1 + 0.5 * (a0 - a1) * (1 + cos(π * t))
# Final bias
bias = alpha * bias_asc + (1 - alpha) * bias_alibi
```

- **목적**: 검증된 positional bias(ALiBi)와 ASC를 선형 혼합
- **장점**:
  - 초기 학습 안정성 향상 (ALiBi 비중 높음)
  - 점진적 ASC 비중 증가로 부드러운 전환
- **스케줄**: Cosine annealing
- **실제 사용**: `[0.30, 0.70]` (초반 ALiBi 70% → 후반 ASC 70%)
- **설정**: `asc_cfg.use_alibi_mix`, `asc_cfg.alpha_start/end/schedule`

### 1.6 Hard Limiter (강제 제한기)
**위치**: `ascender_bias.py:68-70, 494-509`

```python
hard_max_ratio: float = 0.85       # 절대 상한
hard_target_ratio: float = 0.55    # 제한 시 목표값
```

#### 동작
```python
ratio_now = std(bias) / std(scores)
if ratio_now > hard_max_ratio:
    scale_factor = hard_target_ratio / ratio_now
    bias *= scale_factor  # In-place rescale
```

- **목적**: Auto-calibration이 실패해도 폭주 방지 (최후의 안전장치)
- **조건**: `bias/score 비율 > 0.85` 시 발동
- **동작**: 0.55로 강제 스케일링
- **특징**: Gradient 영향 없음 (detach)
- **설정**: `asc_cfg.hard_max_ratio`, `asc_cfg.hard_target_ratio`

### 1.7 Std-Match Normalization
**위치**: `transformer.py:377-383`

```python
if getattr(self, "enable_std_match", True):
    b_std = self._masked_std_bias(runtime_bias, attn_mask)
    r = float(getattr(self, "std_match_ratio", 1.0))
    runtime_bias = (runtime_bias / b_std) * (t_std * r)
else:
    cap = float(getattr(self, "bias_softcap", 6.0))
    runtime_bias = runtime_bias.clamp(min=-cap, max=cap)
```

#### Per-head vs Global
```python
per_head_mode = (self.biaser is not None) and (
    getattr(self.biaser.cfg, "per_head_scale", False) or
    getattr(self.biaser.cfg, "per_head_gate", False)
)
scores_std = self._masked_std_scores(scores, attn_mask, per_head=per_head_mode)
```

- **목적**: 바이어스 크기를 attention scores의 std에 맞춰 정규화
- **공식**: `bias_normalized = (bias / std(bias)) * (std(scores) * r)`
- **r 값**: std_match_ratio (0.30 권장)
- **per-head 모드**: head별 독립적 정규화 가능
- **fallback**: 비활성화 시 `bias_softcap` 사용
- **설정**: `model.std_match_ratio_override`

### 1.8 NaN/Inf Guard
**위치**: 여러 곳

```python
# ascender_bias.py:411
bias = torch.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)

# ascender_bias.py:431
if not torch.isfinite(scaled).all():
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

# transformer.py:325, 357
scores = torch.nan_to_num(scores, nan=0.0, posinf=80.0, neginf=-80.0).clamp(-80, 80)
runtime_bias = torch.nan_to_num(runtime_bias, nan=0.0, posinf=80.0, neginf=-80.0)
```

- **목적**: 수치 오버플로우/언더플로우 방지
- **적용 지점**:
  1. Raw bias 생성 직후
  2. γ-scale 적용 후
  3. Scores/bias softmax 직전
- **전략**: NaN/Inf → 안전한 값으로 치환

---

## 2. 부가 요소

### 2.1 Per-Head Parameters (헤드별 독립 파라미터)
**위치**: `ascender_bias.py:50-51, 56-57, 305-347`

```python
per_head_scale: bool = False  # γ per-head
per_head_gate: bool = False   # σ per-head
```

#### Lazy Initialization
```python
def _ensure_gamma_init(self, h: int, device):
    if self.cfg.per_head_scale:
        base = math.log(global_scale_init)
        jitter = torch.randn(h, device=device) * jitter_std
        self.gamma_log = nn.Parameter(torch.full((h,), base) + jitter)
```

- **목적**: 각 attention head가 독립적인 bias 강도 학습
- **파라미터**:
  - `gamma_log`: (H,) - head별 스케일
  - `gate_param`: (H,) - head별 게이트
- **Jitter**: symmetry breaking을 위한 작은 노이즈 추가
- **설정**: `asc_cfg.per_head_scale`, `asc_cfg.per_head_gate`

### 2.2 Runtime Control Lock
**위치**: `transformer.py:200-222`, `ascender_bias.py:205-209`

```python
def lock_runtime_controls(self):
    """Freeze current runtime hyperparameters"""
    self._locked_runtime = True
    self._init_runtime_snapshot = (
        float(self.attn_temperature),
        float(self.sparsify_k_frac),
        float(self.std_match_ratio),
        float(self.v_gain_epsilon),
    )
```

- **목적**: 실험 중 하이퍼파라미터 변경 방지
- **잠금 항목**: τ, topk, r, v_gain_epsilon
- **복원**: 매 forward() 시작 시 스냅샷 값으로 복원
- **배포 모드**: γ/σ gradient도 동결 가능

### 2.3 Drift Warning (드리프트 경고)
**위치**: `ascender_bias.py:178-235`

```python
def _maybe_drift_warn(self):
    """Runtime과 expected 값 불일치 시 1회 경고"""
    if drift:
        print(f"[ASC RUNTIME DRIFT][{self.role}] expected (...) but got (...)")
        self.drift_warned_once.fill_(1)
```

- **목적**: Config와 실제 runtime 값 불일치 디버깅
- **모니터링**: τ, topk, r, v_gain_epsilon
- **특징**: 한 번만 경고 (로그 중복 방지)
- **weakref 사용**: 순환 참조 방지

### 2.4 Sparsify K-Frac (키 차원 희소화)
**위치**: `transformer.py:157, 366-368`

```python
sparsify_k_frac: float = 0.0  # 0.0~1.0
# ...
if 0.0 < k_frac < 1.0:
    runtime_bias = self._sparsify_last_dim(runtime_bias, k_frac=k_frac)
```

#### 구현
```python
@staticmethod
def _sparsify_last_dim(bias: torch.Tensor, k_frac: float, use_abs: bool = True):
    k = max(1, int(S * k_frac))
    sel = bias.abs() if use_abs else bias
    topv, topi = torch.topk(sel, k, dim=-1)
    mask = torch.zeros_like(bias, dtype=torch.bool).scatter_(-1, topi, True)
    return torch.where(mask, bias, torch.zeros_like(bias))
```

- **목적**: 각 query에 대해 상위 k% key만 bias 적용 (나머지 0)
- **선택 기준**: |bias| 값 기준 top-k
- **효과**:
  - 계산 효율성 향상
  - Long-range dependency 선택적 강조
- **기본값**: 0.0 (비활성화)
- **설정**: `self.sparsify_k_frac`

### 2.5 V-Gain Epsilon (Value 경로 게인)
**위치**: `transformer.py:158, 466-477`

```python
v_gain_epsilon: float = 0.0
# ...
if v_gain_epsilon > 0.0 and attn_bias is not None:
    with torch.no_grad():
        m = attn_bias.abs().mean(dim=2)  # (B,H,S)
        m_norm = (m / (m.mean(dim=-1, keepdim=True) + 1e-6)).unsqueeze(-1)
        gain = 1.0 + v_gain_epsilon * m_norm
    vh = vh * gain.detach()
```

- **목적**: Bias가 강한 key의 value를 미세 증폭
- **계산**:
  1. key별 평균 bias 계산
  2. 정규화 (전체 평균 대비)
  3. gain = 1 + ε × normalized_bias
- **범위**: gain ∈ [1, 1+ε]
- **특징**: Gradient 영향 없음
- **기본값**: 0.0 (비활성화)
- **설정**: `self.v_gain_epsilon`

### 2.6 Attention Temperature (어텐션 온도)
**위치**: `transformer.py:156, 320-323`

```python
attn_temperature: float = 1.0
# ...
tau = float(getattr(self, "attn_temperature", 1.0))
if tau != 1.0:
    scores = scores / tau
```

- **목적**: Attention distribution의 sharpness 조절
- **효과**:
  - τ > 1: 더 균등한 분포 (smoothing)
  - τ < 1: 더 날카로운 분포 (sharpening)
- **적용**: Bias 추가 이전, scores에만 적용
- **기본값**: 1.0 (no temperature)
- **설정**: `self.attn_temperature`

### 2.7 Batch Std Aggregation
**위치**: `ascender_bias.py:87-88`, `transformer.py:378-380`

```python
std_batch_mean: bool = True
# ...
if self.biaser.cfg.std_batch_mean:
    b_std = b_std.mean(dim=0, keepdim=True)  # (1,H,1,1)
```

- **목적**: 배치 간 bias 크기 일관성 유지
- **효과**: Batch variance 감소
- **기본값**: True
- **설정**: `asc_cfg.std_batch_mean`

### 2.8 Residual Scale (잔차 스케일링)
**위치**: `transformer.py:513, 541, 765`

```python
self.resid1_scale: float = 1.0
# ...
x = x + self.resid1_scale * self.dropout1(attn_out)
```

- **목적**: Aggressive bias 사용 시 residual connection 안정화
- **적용**: Biaser 부착된 layer만 0.9로 조정
- **효과**:
  - Gradient flow 안정화
  - 과도한 bias 영향 완화
- **기본값**: 1.0
- **조정값**: 0.9 (biaser 있는 경우)

---

## 3. 아키텍처 확장 옵션

### 3.1 Residual Bias Path (잔차 바이어스 경로)
**파일**: `architectural_mods.py:25-106`, `transformer.py:169-178, 404-433`

#### 개념
```
Normal Path:    softmax(scores) → attn_normal
Biased Path:    softmax(scores + bias) → attn_biased
Final Output:   α·attn_normal + (1-α)·attn_biased
```

#### 구현
```python
enable_residual_path: bool = False  # Config
alpha_logit = nn.Parameter(torch.zeros(n_heads))  # Per-head α

# Forward
attn_normal = softmax(scores)
attn_biased = softmax(scores + bias)
alpha = sigmoid(alpha_logit).view(1, -1, 1, 1)
attn = alpha * attn_normal + (1 - alpha) * attn_biased
```

#### 특징
- **장점**: Bias가 softmax normalization에 압도되지 않음
- **학습**: α는 head별로 학습되어 최적 믹싱 비율 자동 결정
- **안정성**: 항상 normal path와 혼합되어 안전
- **활성화**: `enable_residual_path: true` in config
- **적용 범위**: Encoder/Decoder 모든 attention에 가능

### 3.2 Gated Bias Integration
**파일**: `architectural_mods.py:112-185`

#### 개념
Query 특성 기반으로 bias 신뢰도를 학습

```python
gate = σ(W_gate @ q)  # (B, T, H)
final_bias = gate * structural_bias
```

#### 특징
- **Query별 적응**: 각 위치마다 다른 bias 강도
- **학습 가능**: 언제 bias를 신뢰할지 데이터로부터 학습
- **복잡도**: 추가 네트워크 필요 (d_model → d_model/4 → n_heads)

### 3.3 Multi-Scale Bias
**파일**: `architectural_mods.py:190-272`

#### 개념
서로 다른 스케일의 bias를 독립적으로 적용

```
Local (σ=2):   근거리 토큰 cohesion
Mid (σ=8):     중거리 패턴
Global:        전역 alignment
```

#### 구현
```python
bias_total = w_local·bias_local + w_mid·bias_mid + w_global·bias_global
```

#### 특징
- **스케일 분리**: 각 거리 범위에 최적화된 bias
- **학습 가능**: 스케일별 가중치 학습
- **유연성**: 필요한 스케일만 활성화 가능

### 3.4 Bias-Conditioned Value
**파일**: `architectural_mods.py:278-353`

#### 개념
Bias가 attention distribution뿐 아니라 value도 modulation

```python
bias_key = bias.mean(dim=2)  # (B, H, S)
v_mod = 1.0 + ε * tanh(bias_key)
vh = vh * v_mod
```

#### 특징
- **양방향 영향**: WHERE + WHAT
- **미세 조정**: ε로 modulation 강도 제어
- **구조 정보**: Bias가 feature 중요도에 반영

### 3.5 Hierarchical Bias (계층적 바이어스)
**파일**: `architectural_mods.py:359-443`

#### 개념
Coarse-to-fine 2단계 refinement

```
Stage 1 (Coarse):  Positional bias로 rough pattern 형성
Stage 2 (Fine):    Coarse attention으로 가중된 content bias 추가
```

#### 구현
```python
# Stage 1
scores_coarse = scores + strength_coarse * bias_positional
attn_coarse = softmax(scores_coarse)

# Stage 2
bias_fine = bias_content * attn_coarse.detach()  # Weighted by coarse
scores_final = scores_coarse + strength_fine * bias_fine
```

#### 특징
- **위치 + 내용**: 구조와 의미 정보 분리
- **점진적 정제**: Coarse가 fine의 가이드
- **효율성**: 이미 주목하는 영역만 fine-tuning

---

## 4. 비교 지표 체계

### 4.1 성능 지표 (Performance Metrics)

#### Primary Metrics
| 지표 | 설명 | 범위/단위 | 목표 |
|------|------|----------|------|
| **Loss** | Cross-entropy loss | [0, ∞) | ↓ |
| **Perplexity (PPL)** | exp(loss) | [1, ∞) | ↓ |
| **Accuracy** | Token-level 정확도 | [0, 100]% | ↑ |

#### Computational Metrics
| 지표 | 설명 | 단위 | 비고 |
|------|------|------|------|
| **Train Time** | Epoch당 학습 시간 | seconds | Baseline 대비 비율 |
| **Memory** | Peak GPU memory | GB | Batch size 동일 ���건 |
| **Parameters** | 추가 파라미터 수 | K/M | γ, σ, α 등 |

### 4.2 안정성 지표 (Stability Metrics)

#### Bias-Score Relationship
| 지표 | 공식 | 의미 | 이상적 범위 |
|------|------|------|------------|
| **Bias Ratio (r)** | std(bias) / std(scores) | Bias 상대적 강도 | 0.23~0.35 |
| **Bias Mean** | mean(bias) | Bias 중심 | ~0 (centering off) |
| **Bias Std** | std(bias) | Bias 변동성 | Layer별 다름 |
| **Scores Std** | std(scores) | Attention 변동성 | Layer별 다름 |

#### Component Metrics
| 지표 | 범위 | 의미 | 위치 |
|------|------|------|------|
| **γ_effective** | [gamma_min, gamma_cap] | 실제 scale 강도 | AscenderBias |
| **σ_effective** | [gate_floor, gate_ceiling] | 실제 gate 열림 | AscenderBias |
| **α_alibi** | [alpha_start, alpha_end] | ASC vs ALiBi 비율 | 스케줄링 |
| **α_residual** | [0, 1] | Normal vs Biased 믹싱 | 아키텍처 모드 |

### 4.3 진단 지표 (Diagnostic Metrics)

#### Attention Quality
```python
# Entropy (집중도)
H = -Σ(p·log(p))  # 낮을수록 집중적

# Uniformity (균등성)
U = max(p) / mean(p)  # 높을수록 편중

# Locality (국소성)
L = Σ|i-j|·p_ij  # 낮을수록 가까운 토큰 주목
```

#### Gradient Health
| 지표 | 의미 | 계산 |
|------|------|------|
| **Grad Norm** | Gradient 크기 | `torch.nn.utils.clip_grad_norm_` |
| **Grad Variance** | Layer별 gradient 차이 | Per-layer std |
| **Dead Neurons** | 활성화 0인 비율 | ReLU 등 |

### 4.4 비교 프로토콜

#### Baseline Configurations
1. **Vanilla Transformer**: ASCender off
2. **ALiBi Only**: Positional bias만
3. **ASCender Safe**: Conservative settings
4. **ASCender Aggressive**: High impact settings

#### Ablation Studies
| 제거 요소 | 목적 |
|----------|------|
| Auto-calibration | 자동 보정 효과 |
| ALiBi Mix | 혼합 전략 효과 |
| Gate Ceiling | 상한 제한 필요성 |
| Per-head params | Head별 독립성 효과 |
| Residual path | 아키텍처 수정 효과 |

#### Measurement Protocol
```yaml
# 공정 비교 조건
- Same random seed (42)
- Same batch size (4)
- Same sequence length (256)
- Same vocab size (30K)
- Same architecture (d=256, h=8, L=3)
- Same optimizer (AdamW)
- Same data (WikiText-2)
- 3회 반복 후 평균±std
```

---

## 5. 코드 위치 참조

### 5.1 핵심 파일 구조

```
src/models/
├── transformer.py              # Main model & MHA
├── ascender_bias.py           # Bias generation logic
└── architectural_mods.py      # Advanced architectures

configs/
├── baseline.yaml              # Vanilla transformer
├── ascender256.yaml          # ASCender main config
├── ascender_safe.yaml        # Conservative preset
└── ascender_very_aggressive.yaml  # Aggressive preset

src/
├── train.py                   # Training loop
└── utils/
    └── sched_ascender.py     # Hyperparameter scheduling
```

### 5.2 주요 클래스/함수 위치

#### Stabilization Elements
| 요소 | 파일 | 라인 | 함수/변수 |
|------|------|------|----------|
| Centering | `ascender_bias.py` | 42, 407-410 | `enable_centering` |
| Clamping | `ascender_bias.py` | 39-40, 412 | `clamp_min/max` |
| Auto-calibration | `ascender_bias.py` | 61-67, 434-488 | `use_auto_calibrate` |
| Gate Ceiling/Floor | `ascender_bias.py` | 57-59, 340-347 | `gate_floor/ceiling` |
| ALiBi Mix | `ascender_bias.py` | 80-86, 414-420 | `use_alibi_mix` |
| Hard Limiter | `ascender_bias.py` | 68-70, 494-509 | `hard_max_ratio` |
| Std-Match | `transformer.py` | 377-383 | `enable_std_match` |
| NaN Guard | Multiple | | `torch.nan_to_num` |

#### Additional Features
| 요소 | 파일 | 라인 | 함수/변수 |
|------|------|------|----------|
| Per-head params | `ascender_bias.py` | 50-51, 56-57 | `per_head_scale/gate` |
| Runtime Lock | `transformer.py` | 200-222 | `lock_runtime_controls()` |
| Drift Warning | `ascender_bias.py` | 211-235 | `_maybe_drift_warn()` |
| Sparsify K | `transformer.py` | 269-277, 366-368 | `_sparsify_last_dim()` |
| V-Gain | `transformer.py` | 466-477 | `v_gain_epsilon` |
| Temperature | `transformer.py` | 320-323 | `attn_temperature` |
| Batch Std | `ascender_bias.py` | 87-88 | `std_batch_mean` |
| Residual Scale | `transformer.py` | 513, 541, 765 | `resid1_scale` |

#### Architectural Extensions
| 확장 | 파일 | 라인 | 클래스 |
|------|------|------|--------|
| Residual Bias Path | `architectural_mods.py` | 25-106 | `MultiHeadAttentionWithResidualBias` |
| Residual Path (Integrated) | `transformer.py` | 169-178, 404-433 | `enable_residual_path` |
| Gated Integration | `architectural_mods.py` | 112-185 | `GatedBiasAttention` |
| Multi-Scale | `architectural_mods.py` | 190-272 | `MultiScaleBiasAttention` |
| Value Conditioning | `architectural_mods.py` | 278-353 | `BiasConditionedValueAttention` |
| Hierarchical | `architectural_mods.py` | 359-443 | `HierarchicalBiasAttention` |

### 5.3 Config 매핑

#### 안정화 요소 Config
```yaml
asc_cfg:
  # Stabilization
  enable_centering: false           # [ascender_bias.py:42]
  clamp_min: -2.0                   # [ascender_bias.py:39]
  clamp_max: 2.0                    # [ascender_bias.py:40]
  use_auto_calibrate: true          # [ascender_bias.py:62]
  target_ratio: 0.30                # [ascender_bias.py:63]
  gate_floor: 0.20                  # [ascender_bias.py:58]
  gate_ceiling: 0.70                # [ascender_bias.py:59]
  use_alibi_mix: true               # [ascender_bias.py:81]
  alpha_start: 0.30                 # [ascender_bias.py:82]
  alpha_end: 0.70                   # [ascender_bias.py:83]
  hard_max_ratio: 0.85              # [ascender_bias.py:69]
  hard_target_ratio: 0.55           # [ascender_bias.py:70]

  # Additional
  per_head_scale: true              # [ascender_bias.py:50]
  per_head_gate: false              # [ascender_bias.py:56]
  std_batch_mean: true              # [ascender_bias.py:88]
  gamma_min: 0.30                   # [ascender_bias.py:72]
  gamma_cap: 4.0                    # [ascender_bias.py:73]

model:
  std_match_ratio_override: 0.30    # [transformer.py:679]
  enable_residual_path: false       # [transformer.py:675]
```

---

## 6. 요약 테이블

### 6.1 안정화 우선순위

| 순위 | 요소 | 중요도 | 기본값 권장 | 비고 |
|------|------|--------|------------|------|
| 1 | **Std-Match** | ★★★★★ | ON | 가장 핵심적 |
| 2 | **Gate Ceiling/Floor** | ★★★★☆ | [0.20, 0.70] | 폭주 방지 |
| 3 | **Hard Limiter** | ★★★★☆ | 0.85 / 0.55 | 최후 안전망 |
| 4 | **Clamping** | ★★★☆☆ | [-2, 2] | γ 이전 보호 |
| 5 | **Auto-Calibration** | ★★★☆☆ | ON (aggressive) | 자동 조정 |
| 6 | **ALiBi Mix** | ★★☆☆☆ | Optional | 초기 안정성 |
| 7 | **Centering** | ★☆☆☆☆ | OFF | 디버깅용만 |

### 6.2 부가 기능 활용도

| 기능 | 성능 영향 | 복잡도 | 추천 시나리오 |
|------|----------|--------|--------------|
| **Per-head params** | ↑↑ | Low | Always |
| **V-Gain Epsilon** | ↑ | Low | Long sequences |
| **Sparsify K** | → | Medium | Efficiency 중시 |
| **Temperature** | → | Low | Fine-tuning |
| **Residual Path** | ↑↑↑ | Medium | Aggressive bias |
| **Runtime Lock** | - | Low | Deployment |
| **Drift Warning** | - | Low | Development |

### 6.3 아키텍처 확장 비교

| 아키텍처 | 복잡도 | 파라미터 증가 | 성능 | 안정성 | 사용 예 |
|----------|--------|--------------|------|--------|---------|
| **Standard (Additive)** | Low | +0 | Baseline | Medium | Default |
| **Residual Path** | Medium | +H (α) | ↑↑ | High | Aggressive |
| **Gated Integration** | High | +d²/4 | ↑↑ | High | Adaptive |
| **Multi-Scale** | High | +3×biaser | ↑↑↑ | Medium | Multi-domain |
| **Value Conditioning** | Medium | +1 (ε) | ↑ | Medium | Semantic |
| **Hierarchical** | High | +2×biaser | ↑↑ | High | Compositional |

---

## 7. 베스트 프랙티스

### 7.1 초기 설정 (Getting Started)

```yaml
# Conservative & Stable
asc_cfg:
  clamp_min: -2.0
  clamp_max: 2.0
  gate_floor: 0.25
  gate_ceiling: 0.65
  use_auto_calibrate: false
  use_alibi_mix: true
  alpha_start: 0.5
  alpha_end: 0.6
  per_head_scale: false

model:
  std_match_ratio_override: 0.25
  enable_residual_path: false
```

### 7.2 공격적 설정 (Aggressive)

```yaml
# High-Impact (requires Residual Path)
asc_cfg:
  clamp_min: -2.0
  clamp_max: 2.0
  gate_floor: 0.20
  gate_ceiling: 0.70
  use_auto_calibrate: true
  target_ratio: 0.30
  use_alibi_mix: true
  alpha_start: 0.30
  alpha_end: 0.70
  per_head_scale: true

model:
  std_match_ratio_override: 0.30
  enable_residual_path: true  # REQUIRED
```

### 7.3 디버깅 체크리스트

1. **Loss 폭발 시**:
   - [ ] `hard_max_ratio` 확인 (< 0.85?)
   - [ ] `gamma_cap` 확인 (< 6.0?)
   - [ ] `enable_residual_path` 고려
   - [ ] `std_match_ratio` 낮추기

2. **Bias 효과 없을 시**:
   - [ ] `gate_floor` 확인 (> 0.15?)
   - [ ] `std_match_ratio` 확인 (> 0.2?)
   - [ ] `gamma_effective` 로그 확인
   - [ ] Per-head params 활성화

3. **학습 불안정 시**:
   - [ ] `use_auto_calibrate` OFF
   - [ ] `resid1_scale` 낮추기 (0.9 → 0.8)
   - [ ] `gate_ceiling` 낮추기
   - [ ] Residual Path 활성화

---

## 부록: 코드 스니펫

### A. Bias Ratio 추출 (로깅)
```python
# In training loop
if step % 100 == 0:
    mha = model.decoder.layers[0].self_attn
    if hasattr(mha, 'attn_bias') and hasattr(mha, 'attn_pre'):
        bias_std = mha.attn_bias.std().item()
        scores_std = mha.attn_pre.std().item()
        ratio = bias_std / (scores_std + 1e-6)
        print(f"[L0] Bias Ratio: {ratio:.3f}")
```

### B. γ/σ 모니터링
```python
biaser = model.decoder.layers[0].biaser_self
if biaser:
    gamma_eff = biaser.gamma_effective  # Property
    gate_eff = biaser.gate_effective    # Property
    print(f"γ={gamma_eff:.3f}, σ={gate_eff:.3f}")
```

### C. α 추출 (Residual Path)
```python
if model.cfg.enable_residual_path:
    mha = model.decoder.layers[0].self_attn
    if hasattr(mha, '_alpha_effective'):
        alpha_per_head = mha._alpha_effective  # (H,)
        print(f"α (normal/biased mix): {alpha_per_head.mean():.3f}")
```

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-11-06
**작성자**: Claude
**목적**: 슬라이드 작성 참고자료
