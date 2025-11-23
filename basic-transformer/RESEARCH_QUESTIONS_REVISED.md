# 연구 질문 및 설계 원칙 (실험 결과 반영)

## 제4절 연구 질문

본 연구는 다음 세 가지 질문을 중심으로 전개되었으며, WikiText-103 언어 모델링 실험을 통해 검증하였다.

### RQ1 (구조): 지역 규칙 기반 어텐션 구조 형성
**질문**: 단순한 지역 규칙(Alignment, Separation, Cohesion)이 정확도를 유지하면서도 안정적이고 해석 가능한 어텐션 구조를 유도할 수 있는가?

**실험 결과**:
- ✅ **구조 형성**: 성공
  - Bias 히트맵 분석 결과, 의도한 spatial structure가 명확히 형성됨
  - Separation: 대각선 중심 패턴 확인 (diagonal mean: 1.93 vs off-diagonal: 0.56)
  - Cohesion: 로컬 neighborhood 강화 효과 관찰 (σ=0.91, 충분한 분산)
  - Z-score 정규화 시 해석 가능한 대각선 밴드 구조 확인

- ⚠️ **정확도 유지**: 부분적 성공
  - Best case: +0.01% improvement (A+S 조합: 8.4954 vs baseline: 8.4961)
  - **성능 저하 없음**: 모든 variant가 baseline 대비 ±0.01% 이내
  - **안정적 학습**: 모든 configuration에서 안정적인 수렴 확인

**해석**:
- Boids 기반 지역 규칙은 **구조적 귀납적 편향(structural inductive bias)을 성공적으로 주입**함
- 하지만 **언어 모델링 task에서는 spatial proximity가 linguistic relevance와 직접 대응하지 않음**
- 구조는 형성되지만, 해당 구조가 언어적으로 유의미한지는 task-dependent


### RQ2 (효율성): 계산-성능 효율성 개선
**질문**: 이러한 편향이 불필요한 장거리 상호작용을 줄여 계산–성능 효율성을 개선할 수 있는가?

**실험 결과**:
- ❌ **성능 효율성**: 미개선
  - 모든 ASCender variant가 baseline과 유사한 loss (±0.01%)
  - 장거리 상호작용 억제가 성능 개선으로 이어지지 않음
  - Encoder bias만으로는 decoder 예측 성능에 직접적 영향 제한적

- ⚠️ **계산 효율성**: 미측정
  - 현재 실험에서는 학습 시간, FLOPs, 메모리 사용량 측정 안 함
  - Bias 계산 자체는 추가 연산 발생 (가우시안 커널, QK 유사도)
  - 실제 효율성 개선을 위해서는 **sparse attention**과 결합 필요

**해석**:
- Bias를 추가해도 attention은 여전히 full sequence에 대해 계산됨
- **실제 계산량 감소 없음**: softmax 전 bias만 추가 (O(n²) 복잡도 유지)
- 효율성 개선을 위해서는:
  1. Top-k attention (k << n)
  2. Sliding window attention
  3. Kernel-based linear attention
  - 등의 **sparse mechanism과 결합** 필요


### RQ3 (일반화): 견고성 및 민감도 분석
**질문**: 성능 개선이 과제와 규모 전반에 걸쳐 견고하게 유지되며, 편향 강도 및 이웃 정의 방식에 얼마나 민감한가?

**실험 결과**:

#### 3-1. Component 조합별 견고성
| Configuration | Final Loss | Relative to Baseline | Consistency |
|---------------|------------|---------------------|-------------|
| Baseline | 8.4961 | - | - |
| A+S | 8.4954 | +0.01% | ✅ Best |
| S | 8.4954 | +0.01% | ✅ Best |
| A | 8.4956 | +0.01% | ✅ Good |
| C | 8.4958 | +0.00% | ✅ Good |
| S+C (n=3) | 8.4959±0.0001 | +0.00% | ✅ Robust |
| A+C | 8.4961 | -0.00% | ✅ Neutral |
| A+S+C | 8.4962 | -0.00% | ✅ Neutral |

- ✅ **일관성**: 매우 높음
  - S+C 조합 3회 반복 실험: std=0.0001 (매우 낮은 분산)
  - 모든 variant가 stable convergence

- ⚠️ **성능 변동**: 매우 작음
  - 최대 차이: 0.0008 (0.01%)
  - **통계적으로 유의미하지 않음** (실험 noise 범위 내)

#### 3-2. Hyperparameter 민감도

**현재 설정**:
```yaml
w_align: 3.5
w_sep: 1.5
w_coh: 5.0
sigma_sep: 2.0
sigma_coh: 50.0
```

**분석**:
- ⚠️ **σ_coh=50.0의 문제점**:
  - 시퀀스 길이(256)의 20%에 해당하는 매우 넓은 kernel
  - 대부분의 토큰이 유사한 cohesion bias → 차별화 효과 감소
  - Diagnostic 분석: T=256일 때 diagonal vs off-diagonal 차이는 크지만, gradient가 너무 완만함

- **Component weight 균형**:
  - Separation (w=1.5)이 단독으로 가장 효과적
  - Cohesion (w=5.0)이 가장 크지만 sigma가 너무 커서 효과 희석
  - Alignment (w=3.5)은 QK 유사도 기반이라 task-dependent

**민감도 결론**:
- ✅ **Component 선택에 견고함**: A/S/C 어떤 조합이든 성능 유지
- ⚠️ **Kernel width에 민감함**: σ_coh가 너무 크면 효과 감소
- 💡 **제안**: σ_coh = 8~12로 줄이면 더 명확한 local structure 가능

#### 3-3. Task 일반화 (제한적)
- ❌ **현재**: WikiText-103 (언어 모델링)만 테스트
- 💡 **필요**: 다양한 task에서 검증 필요
  - Vision-Language: Image captioning, VQA (spatial structure가 더 중요)
  - Structured NLP: Dependency parsing, NER (문법 구조 명확)
  - Long-context: 문서 요약, QA (장문에서 효과 기대)

---

## 제5절 설계 원칙 (실험 결과 기반 개정)

ASCender는 세 가지 원칙을 따르며, WikiText-103 실험을 통해 각 원칙의 유효성을 검증하였다.

### (P1) 호환성 (Compatibility) ✅ **검증됨**
**원칙**: 어떤 어텐션 계층에도 모듈식으로 적용 가능

**구현**:
- Encoder/Decoder self-attention, cross-attention 모두 지원
- Layer-wise 선택적 적용 (L0, L1만 활성화 등)
- Per-head 독립적인 γ scaling 지원

**검증 결과**:
- ✅ Encoder 3개 layer에 일관되게 적용 성공
- ✅ Decoder에도 동일한 방식으로 적용 가능 확인 (diagnostic 테스트)
- ✅ Residual Bias Path와도 호환 가능 (enable_residual_path 옵션)

**제한사항**:
- Causal masking과의 상호작용 주의 필요 (decoder self-attention)
- Cross-attention에서는 source/target sequence 길이 불일치 고려 필요


### (P2) 약한 사전지식 (Weak Priors) ✅ **부분 검증**
**원칙**: 부드럽고 학습 가능한 계수로 제어

**구현**:
- Learnable γ (per-head scale): 각 head가 bias 강도 조절
- Learnable gate: bias on/off 제어 (floor=0.1, ceiling=0.9)
- Soft gaussian kernels: 하드한 cutoff 대신 부드러운 감쇠
- Clamp range: [-12, 12]로 충분히 넓게 설정

**검증 결과**:
- ✅ **부드러운 bias**: σ=0.91로 충분한 분산, 학습 가능
- ✅ **성능 저하 없음**: 모든 variant가 baseline ±0.01% 이내
- ⚠️ **학습된 gate 효과 미미**: 최종 성능 차이가 너무 작아 gate의 실제 학습 패턴 분석 어려움

**개선 방향**:
- Gate 학습 궤적 로깅 추가 (초기 vs 후기 epoch)
- Alpha 값 분석으로 model이 실제로 bias를 얼마나 사용하는지 확인
- **더 강한 bias 실험** 필요: w_coh=10~15, σ_coh=8~12


### (P3) 최소 복잡성 (Minimal Complexity) ⚠️ **부분 검증**
**원칙**: 기존 커널을 그대로 활용, 별도 연산 불필요

**구현**:
- 가우시안 커널 재사용 (Separation, Cohesion)
- QK 내적 재사용 (Alignment)
- Attention 계산 흐름에 bias 한 번만 추가: `logits = QK + bias`

**검증 결과**:
- ✅ **구현 단순성**: ~200 lines 코드로 구현 (ascender_bias.py)
- ✅ **기존 Transformer와 호환**: 학습 파이프라인 수정 최소화
- ⚠️ **실제 연산량**: 측정 안 됨

**복잡도 분석**:
```
Bias 계산: O(n²) per layer (n = sequence length)
- Separation: O(n²) gaussian distance
- Cohesion: O(n²) gaussian distance
- Alignment: O(n² × d) QK inner product (이미 attention에서 계산됨, 재사용)

Total overhead per forward pass:
- Encoder 3 layers × n²: 3 × (256)² = ~200K operations
- Negligible compared to full model (d_model=256, d_ff=1024)
```

**실측 필요**:
- Training time (with/without bias)
- Memory usage (peak GPU memory)
- FLOPs count
- Inference throughput

---

## 제6절 연구 질문 재평가 및 수정된 연구 방향

### 실험을 통해 얻은 통찰

1. **Spatial Bias ≠ Linguistic Structure**
   - 물리적 근접성 기반 bias는 언어적 관련성과 직접 대응하지 않음
   - NLP task에서는 **semantic similarity**가 더 중요

2. **Encoder vs Decoder**
   - Encoder bias만으로는 decoder 예측 성능 개선 제한적
   - **Decoder self-attention에 직접 적용** 필요

3. **Task Dependency**
   - 언어 모델링에서는 효과 미미
   - **Vision-Language, Structured Prediction** 등에서 더 유용할 가능성

### 수정된 연구 질문 (RQ v2)

#### RQ1-v2 (Task-Specific Structure)
**질문**: Spatial bias가 효과적인 task의 특성은 무엇인가?

**가설**:
- Spatial locality가 중요한 task (vision, speech)
- 명확한 구조가 있는 task (parsing, POS tagging)
- Long-range dependency보다 local context가 중요한 task

**제안 실험**:
- Image captioning (ViT + GPT)
- Dependency parsing (structured prediction)
- Speech recognition (WavLM + CTC)


#### RQ2-v2 (Adaptive Bias)
**질문**: 고정된 bias 대신 학습 가능한 adaptive bias가 성능을 개선할 수 있는가?

**제안 방향**:
- Learnable σ (layer-wise, head-wise)
- Dynamic kernel width (content-dependent)
- **Residual Bias Path**: α * P_learned + (1-α) * P_biased


#### RQ3-v2 (Efficiency via Sparsity)
**질문**: Bias 기반 sparsification이 실제 계산 효율을 개선할 수 있는가?

**제안 방향**:
- Bias-guided Top-k attention
- Bias threshold 기반 early pruning
- Sparse pattern 학습 후 static pruning

---

## 요약: 연구 질문 답변

| 연구 질문 | 답변 | 근거 |
|----------|------|------|
| RQ1: 구조 형성 | ✅ 성공 (구조), ⚠️ 부분 성공 (정확도) | Heatmap 패턴 명확, 하지만 성능 개선 미미 |
| RQ2: 효율성 | ❌ 미개선 | 계산량 감소 없음, 성능 개선 없음 |
| RQ3: 일반화 | ✅ 견고함 (조합), ⚠️ 민감함 (σ), ❓ 미검증 (task) | S+C std=0.0001, 하지만 σ_coh 영향 큼, 1개 task만 테스트 |

**전체 결론**:
- ASCender는 **구조적 귀납적 편향을 성공적으로 주입**하지만,
- **WikiText-103 언어 모델링에서는 유의미한 성능 개선 없음**
- **Task-dependent**: Vision-language, structured NLP에서 재검증 필요
- **Efficiency 개선을 위해서는 sparsification과 결합** 필요
