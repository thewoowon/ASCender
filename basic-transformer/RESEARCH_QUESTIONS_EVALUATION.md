# 연구 질문 및 설계 원칙 평가 (실험 결과 기반)

## 원본 연구 질문

### RQ1 (구조)
**원본**: 단순한 지역 규칙이 정확도를 유지하면서도 안정적이고 해석 가능한 어텐션 구조를 유도할 수 있는가?

#### 실험 결과 대조

| 평가 항목 | 예측 | 실험 결과 | 판정 |
|----------|------|----------|------|
| **지역 규칙 적용** | 단순한 규칙 (A/S/C) | ✅ 3개 component 성공적 구현 | ✅ 달성 |
| **정확도 유지** | 성능 저하 없음 | ✅ 모든 variant ±0.01% 이내 | ✅ 달성 |
| **안정적 구조** | 학습 과정에서 안정성 | ✅ S+C 3회 반복 std=0.0001 | ✅ 달성 |
| **해석 가능성** | 명확한 패턴 형성 | ✅ 대각선 밴드 구조 확인 (diagonal: 1.93 vs off-diagonal: 0.56) | ✅ 달성 |

**평가**: ✅ **RQ1은 완전히 검증됨**
- Bias 히트맵이 명확한 spatial structure 형성
- 정확도 유지 (성능 저하 없음)
- 안정적 학습 확인

**수정 제안**: 질문 자체는 적절하나, "정확도를 유지하면서"를 "정확도에 유의미한 영향을 주지 않으면서"로 수정 권장 (실제로는 개선도 저하도 없었음)

---

### RQ2 (효율성)
**원본**: 이러한 편향이 불필요한 장거리 상호작용을 줄여 계산–성능 효율성을 개선할 수 있는가?

#### 실험 결과 대조

| 평가 항목 | 예측 | 실험 결과 | 판정 |
|----------|------|----------|------|
| **장거리 상호작용 감소** | Local bias로 long-range 억제 | ⚠️ 미측정 (attention entropy 분석 필요) | ❓ 미검증 |
| **계산 효율성** | 연산량 감소 | ❌ O(n²) 유지, bias 계산 추가 오버헤드 | ❌ 미달성 |
| **성능 효율성** | 더 적은 연산으로 동등/더 나은 성능 | ❌ 성능 개선 없음 (+0.01%) | ❌ 미달성 |

**평가**: ❌ **RQ2는 검증 실패**
- **계산 효율**: Bias 추가로 오히려 연산량 증가 가능
- **성능 효율**: Loss 개선 없음 (±0.01%)
- **장거리 억제**: 실제로 측정하지 않음 (attention pattern entropy 분석 필요)

**핵심 문제**:
- Bias를 추가해도 여전히 **full attention 계산** (O(n²))
- Sparsification이 없으면 효율성 개선 불가능
- "불필요한 장거리 상호작용"의 정의가 명확하지 않음 (언어 모델링에서는 long-range도 중요할 수 있음)

**수정 제안**: RQ2를 2개로 분리
- **RQ2-A (구조적 효과)**: "편향이 어텐션 패턴을 로컬라이즈하여 장거리 의존성을 줄일 수 있는가?"
- **RQ2-B (실제 효율성)**: "로컬라이즈된 패턴을 sparse attention과 결합하여 계산 효율을 개선할 수 있는가?"

---

### RQ3 (일반화)
**원본**: 성능 개선이 과제와 규모 전반에 걸쳐 견고하게 유지되며, 편향 강도 및 이웃 정의 방식에 얼마나 민감한가?

#### 실험 결과 대조

| 평가 항목 | 예측 | 실험 결과 | 판정 |
|----------|------|----------|------|
| **과제 전반** | 다양한 task에서 효과 | ❌ WikiText-103 언어 모델링만 테스트 | ❓ 미검증 |
| **규모 전반** | 다양한 모델 크기 | ❌ d_model=256, n_layers=3 단일 크기 | ❓ 미검증 |
| **성능 개선 견고성** | 일관된 개선 | ⚠️ 개선이 거의 없음 (+0.01%) | ⚠️ 부분 달성 |
| **편향 강도 민감도** | w_align, w_sep, w_coh 영향 | ⚠️ 7개 조합 테스트, 모두 비슷한 결과 | ✅ 견고함 확인 |
| **이웃 정의 민감도** | σ_sep, σ_coh 영향 | ⚠️ 고정값 사용 (σ_coh=50.0 문제 발견) | ⚠️ 부분 검증 |

**평가**: ⚠️ **RQ3는 부분적으로만 검증됨**

**검증된 부분**:
- ✅ **Component 조합 견고성**: A/S/C 어떤 조합이든 비슷한 성능 (±0.0008)
- ✅ **학습 안정성**: S+C 3회 반복 시 std=0.0001

**미검증 부분**:
- ❌ **Task 일반화**: 언어 모델링 외 task 미테스트
- ❌ **규모 일반화**: 단일 모델 크기만 사용
- ⚠️ **Hyperparameter 민감도**: σ만 일부 진단 (고정값으로 학습)

**핵심 문제**:
- "성능 개선"이 전제되어 있지만, 실제로는 개선이 거의 없음
- σ_coh=50.0이 너무 크다는 것을 diagnostic으로 발견했지만, ablation study 안 함

**수정 제안**: RQ3를 더 구체적으로
- **RQ3-A (일관성)**: "편향이 다양한 component 조합에서 안정적으로 작동하는가?"
- **RQ3-B (민감도)**: "성능이 kernel width (σ), bias strength (w), gate 설정에 얼마나 민감한가?"
- **RQ3-C (전이성)**: "편향의 효과가 다른 task (vision-language, structured NLP) 및 모델 규모로 전이되는가?"

---

## 설계 원칙 평가

### P1: 호환성 (Compatibility)
**원본**: 어떤 어텐션 계층에도 모듈식으로 적용 가능

#### 실험 결과 검증

| 검증 항목 | 설계 의도 | 실험 구현 | 판정 |
|----------|----------|----------|------|
| **모듈식 적용** | Plug-and-play | ✅ `asc_bias_enc`, `asc_bias_dec_self` 독립 제어 | ✅ 달성 |
| **다양한 layer** | Encoder/Decoder/Cross-attn | ✅ 모두 지원 (config로 선택) | ✅ 달성 |
| **기존 코드 영향 최소** | Attention 계산 흐름 유지 | ✅ `logits = QK + bias` 한 줄 추가 | ✅ 달성 |

**평가**: ✅ **P1은 완전히 달성됨**

**증거**:
```python
# src/models/transformer.py
if self.biaser is not None:
    bias = self.biaser(q, k, pre_q=x, pre_k=x)
    attn_logits = attn_logits + bias  # 한 줄 추가
```

**실제 사용**:
- Encoder 3 layers: ✅ 작동 확인
- Decoder 2 layers: ✅ 작동 확인 (diagnostic test)
- Residual path: ✅ 호환 가능 (`enable_residual_path` 옵션)

**수정 불필요**: P1은 그대로 유지

---

### P2: 약한 사전지식 (Weak Priors)
**원본**: 부드럽고 학습 가능한 계수로 제어

#### 실험 결과 검증

| 검증 항목 | 설계 의도 | 실험 구현 | 판정 |
|----------|----------|----------|------|
| **부드러운 bias** | Hard constraint 아님 | ✅ Gaussian kernel (soft), clamp [-12, 12] | ✅ 달성 |
| **학습 가능** | γ, gate 학습 | ✅ per_head_scale=true, use_gate=true | ✅ 달성 |
| **충분한 분산** | 학습 신호 제공 | ✅ σ=0.91 (충분한 분산 확인) | ✅ 달성 |
| **성능 저하 없음** | 너무 강한 bias 아님 | ✅ 모든 variant ±0.01% | ✅ 달성 |

**평가**: ✅ **P2는 달성됨**

**증거**:
- Bias 통계: mean=0.56, std=0.91, range=[0, 2.73]
- 학습 가능: γ_log, gate_param이 실제로 학습됨 (gradient 흐름 확인)
- 성능 영향: 거의 없음 (weak prior의 증거)

**제한사항**:
- ⚠️ **"학습 가능"의 실효성**: 최종 loss 차이가 너무 작아서 학습된 γ/gate가 실제로 의미 있는지 불명확
- 💡 **제안**: Gate/γ 학습 궤적 로깅 필요 (epoch별 변화 추적)

**수정 제안**: "부드럽고 학습 가능한 계수로 제어하되, **model이 bias를 무시할 수 있도록** gate mechanism 제공"

---

### P3: 최소 복잡성 (Minimal Complexity)
**원본**: 기존 커널을 그대로 활용, 별도 연산 불필요

#### 실험 결과 검증

| 검증 항목 | 설계 의도 | 실험 구현 | 판정 |
|----------|----------|----------|------|
| **기존 커널 재사용** | 새 primitive 없음 | ✅ Gaussian, inner product 사용 | ✅ 달성 |
| **별도 연산 불필요** | Attention과 독립적 | ⚠️ Bias 계산은 추가 연산 (O(n²)) | ⚠️ 부분 달성 |
| **구현 단순성** | 적은 코드 | ✅ ~200 lines (ascender_bias.py) | ✅ 달성 |

**평가**: ⚠️ **P3는 부분적으로 달성됨**

**달성된 부분**:
- ✅ 구현이 간단함 (200 lines)
- ✅ 기존 Transformer 코드 수정 최소화
- ✅ Gaussian, dot product 등 표준 연산만 사용

**미달성 부분**:
- ⚠️ "별도 연산 불필요"는 **부정확**:
  - Separation/Cohesion: Position-based gaussian (O(n²) 계산)
  - Alignment: QK inner product (이미 attention에서 계산하므로 재사용 가능하지만, 현재는 별도 계산)

**실제 복잡도**:
```python
# Separation bias: O(n²)
dist = (i - j)²
bias_sep = -exp(-dist / (2*σ²))

# Cohesion bias: O(n²)
bias_coh = -dist² / (2*σ²)

# Alignment bias: O(n² * d) → 하지만 QK는 이미 attention에서 계산됨
bias_align = QK / temperature
```

**측정 안 된 것**:
- Training time overhead
- Memory usage
- FLOPs count

**수정 제안**: "기존 Transformer 연산(QK 내적, softmax)과 **호환되는 단순한 연산**만 사용하며, 구현 복잡도를 최소화"

---

## 수정된 연구 질문 (실험 결과 반영)

### 제4절 연구 질문 (개정판)

본 연구는 Boids 알고리즘에서 영감을 받은 attention bias가 Transformer 성능에 미치는 영향을 다음 세 가지 측면에서 검증한다.

#### RQ1 (구조적 귀납적 편향)
**질문**: 단순한 지역 규칙(Alignment, Separation, Cohesion)이 **성능에 부정적 영향을 주지 않으면서** 안정적이고 해석 가능한 어텐션 구조를 형성할 수 있는가?

**실험 결과**: ✅ **검증됨**
- Bias 히트맵 분석 결과 명확한 spatial structure 확인 (diagonal mean: 1.93 vs off-diagonal: 0.56)
- 모든 ASCender variant가 baseline 대비 ±0.01% 이내 (성능 저하 없음)
- 학습 안정성 확인 (S+C 3회 반복 시 std=0.0001)

**함의**: Boids 기반 지역 규칙은 구조적 inductive bias를 성공적으로 주입할 수 있으나, WikiText-103 언어 모델링에서는 성능 개선으로 이어지지 않음.

---

#### RQ2 (효율성 및 locality)
**질문**: 이러한 편향이 어텐션 패턴을 **로컬라이즈**하여, sparse attention mechanism과 결합 시 계산 효율성을 개선할 **가능성**이 있는가?

**실험 결과**: ⚠️ **부분 검증 (구조만 확인, 효율성 미측정)**
- **구조적 locality**: ✅ 대각선 중심 패턴 형성 확인
- **실제 계산 효율**: ❓ 미측정 (FLOPs, memory, training time)
- **성능 효율**: ❌ Loss 개선 없음 (+0.01%)

**한계**:
- 현재 구현은 full attention (O(n²)) 유지, bias 계산이 오히려 오버헤드
- Sparsification 미적용 (top-k, sliding window 등 필요)

**함의**: Bias만으로는 효율성 개선 불가능. **실제 효율성 개선을 위해서는**:
1. Bias 기반 top-k attention selection
2. Sliding window + bias combination
3. Kernel-based linear attention with bias

---

#### RQ3 (견고성 및 민감도)
**질문**: 편향의 효과가 (1) 다양한 component 조합에서 **일관적**이며, (2) hyperparameter (σ, w, gate)에 대해 얼마나 **민감**하고, (3) 다른 task 및 규모로 **전이 가능**한가?

**실험 결과**:
- **(1) Component 조합 견고성**: ✅ **검증됨**
  - 7개 조합 모두 비슷한 성능 (최대 차이 0.0008)
  - S+C 3회 반복: std=0.0001 (매우 일관적)

- **(2) Hyperparameter 민감도**: ⚠️ **부분 검증**
  - σ_coh=50.0이 너무 크다는 것을 diagnostic으로 발견
  - 하지만 ablation study 안 함 (다른 σ 값 미테스트)
  - Component weight (w_align, w_sep, w_coh) 영향 미분석

- **(3) Task/규모 전이성**: ❌ **미검증**
  - WikiText-103 언어 모델링만 테스트
  - d_model=256, n_layers=3 단일 규모

**함의**:
- ASCender는 내부적으로 **robust** (조합/반복 실험에서 일관성)
- 하지만 **task-dependent**: 언어 모델링에서는 효과 제한적
- Vision-language, structured NLP 등 다른 domain에서 재검증 필요

---

## 수정된 설계 원칙

### 제5절 설계 원칙 (개정판)

ASCender는 세 가지 원칙을 따르며, WikiText-103 실험을 통해 각 원칙의 유효성을 검증하였다.

#### (P1) 모듈식 호환성 (Modular Compatibility) ✅
**원칙**: 기존 Transformer 구조를 유지하면서, 어떤 attention layer에도 독립적으로 적용 가능

**구현**:
- Encoder/Decoder self-attention, cross-attention 모두 지원
- Config 기반 layer-wise 선택 (`asc_bias_enc`, `asc_bias_dec_self`)
- Attention 계산 흐름에 한 줄만 추가: `logits = QK + bias`

**검증**: ✅ **달성**
- Encoder 3 layers에 성공적 적용
- Residual Bias Path와 호환 가능
- 기존 학습 파이프라인 수정 최소화

---

#### (P2) 약한 학습 가능 사전지식 (Weak Learnable Priors) ✅
**원칙**: 부드럽고 학습 가능한 bias를 제공하되, **model이 무시할 수 있도록** gate mechanism 포함

**구현**:
- Soft gaussian kernels (hard cutoff 없음)
- Per-head learnable scale (γ) 및 gate (σ)
- 충분히 넓은 clamp range [-12, 12]

**검증**: ✅ **달성**
- Bias 통계: mean=0.56, std=0.91 (충분한 분산)
- 성능 저하 없음 (±0.01%)
- Model이 bias를 선택적으로 사용 가능 (gate mechanism)

**제한**: Gate/γ 학습 궤적 분석 미수행 (향후 필요)

---

#### (P3) 연산 효율성 고려 (Computational Simplicity) ⚠️
**원칙**: 표준 연산(Gaussian, dot product)만 사용하며, 구현 복잡도를 최소화. **실제 효율성 개선을 위해서는 sparsification과 결합 필요**

**구현**:
- ~200 lines 코드 (ascender_bias.py)
- 기존 primitive 재사용 (Gaussian, QK inner product)
- Transformer와 독립적인 bias 계산

**검증**: ⚠️ **부분 달성**
- ✅ 구현 단순성 확인
- ⚠️ 연산 오버헤드 존재 (O(n²) bias 계산)
- ❓ 실제 효율성 미측정 (FLOPs, memory, time)

**제한**:
- 현재는 full attention 유지, 효율성 개선 없음
- 실제 효율성을 위해서는 top-k, sliding window 등과 결합 필요

---

## 최종 평가 요약

| 항목 | 원본 주장 | 실험 검증 | 판정 | 수정 필요성 |
|------|----------|----------|------|------------|
| **RQ1** | 구조 형성 + 정확도 유지 | ✅ 구조 형성, ✅ 정확도 유지 | ✅ 검증됨 | 경미한 수정 ("유지" → "영향 없음") |
| **RQ2** | 계산·성능 효율성 개선 | ❌ 개선 없음, ❓ 계산 미측정 | ❌ 검증 실패 | **대폭 수정 필요** (가능성 제시로 변경) |
| **RQ3** | 과제/규모 전반 일반화 | ⚠️ 조합 견고, ❌ task/규모 미검증 | ⚠️ 부분 검증 | **중간 수정** (3개 하위 질문으로 분리) |
| **P1** | 호환성 | ✅ 모든 layer 지원 | ✅ 달성 | 수정 불필요 |
| **P2** | 약한 사전지식 | ✅ Soft bias, 학습 가능 | ✅ 달성 | 경미한 추가 ("무시 가능" 명시) |
| **P3** | 최소 복잡성 | ⚠️ 구현 단순, 연산 오버헤드 | ⚠️ 부분 달성 | **수정 필요** (효율성 한계 명시) |

---

## 권장 수정사항

### 즉시 수정 필요 (High Priority)

1. **RQ2 전면 개편**
   - 현재: "계산-성능 효율성을 개선할 수 있는가?"
   - 수정: "어텐션 패턴을 로컬라이즈하여, sparse mechanism과 결합 시 효율성 개선 **가능성**을 제공하는가?"

2. **RQ3 구체화**
   - 현재: 단일 질문 (너무 광범위)
   - 수정: 3개 하위 질문 (일관성, 민감도, 전이성)

3. **P3 한계 명시**
   - 현재: "별도 연산 불필요"
   - 수정: "표준 연산만 사용하나, 실제 효율성 개선을 위해서는 sparsification 필요"

### 선택적 수정 (Medium Priority)

4. **RQ1 표현 정밀화**
   - "정확도를 유지" → "정확도에 유의미한 영향을 주지 않으면서"

5. **P2 명시성 강화**
   - "model이 bias를 무시할 수 있도록 gate 제공" 추가

---

## 실험 결과 기반 새로운 통찰

실험을 통해 얻은 예상 밖의 발견:

1. **Task Mismatch**: Spatial proximity ≠ Linguistic relevance
   - 언어 모델링에서는 물리적 거리가 의미적 관련성과 무관
   - Vision-language task에서 더 효과적일 가능성

2. **Encoder vs Decoder**: Encoder bias만으로는 제한적
   - 언어 모델링의 핵심은 decoder causal attention
   - Decoder self-attention에 직접 적용 필요

3. **Sigma Sensitivity**: σ_coh=50.0이 너무 큼
   - 시퀀스 길이(256)의 20%에 해당
   - σ_coh=8~12로 줄이면 더 명확한 local structure 가능

4. **Robust but Ineffective**: 내부적으로는 매우 안정적
   - 조합/반복 실험에서 일관된 결과
   - 하지만 성능 개선은 없음 (±0.01%)
