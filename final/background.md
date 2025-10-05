좋습니다 🙆 이제 **4. Background** 부분을 한국어 학술 논문 톤으로 번역해드리겠습니다. 수식, 정의, 해석 모두 깔끔하게 정리했어요.

---

# 4. 배경 (Background)

## 4.1 Transformer와 자기어텐션 (Transformer and Self-Attention)

### 표기법

길이 (n)의 시퀀스가 임베딩 (X!\in!\mathbb{R}^{n\times d})로 주어졌다고 하자. 헤드 (h)에 대해 다음과 같이 투영(projection)을 정의한다:

[
Q_h = XW^Q_h,\quad K_h = XW^K_h,\quad V_h = XW^V_h,\quad W^Q_h,W^K_h,W^V_h!\in!\mathbb{R}^{d\times d_h}.
]

스케일 내적 기반 자기어텐션은 다음과 같다:

[
A_h=\mathrm{softmax}!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}\right)\in\mathbb{R}^{n\times n},\qquad
Y_h=A_hV_h.
]

멀티헤드 어텐션은 ([Y_1;\dots;Y_H]W^O)로 결합된다.

---

### 위치 정보

현대 Transformer는 위치 정보를 절대 위치 임베딩 대신 **상대적(relative)** 또는 **회전(rotary)** 방식으로 부호화한다. 우리는 일반성을 위해 어텐션 로짓에 **가산적 편향 항(additive bias)**을 포함한 형태를 고려한다:

[
\tilde{A}_h=\mathrm{softmax}!\Big(\frac{Q_hK_h^\top}{\sqrt{d_h}} + B_h\Big),
]

여기서 (B_h\in\mathbb{R}^{n\times n})은 학습되거나 계산되는 임의의 바이어스 행렬이다. **ASCender**는 이 (B_h)를 **보이드(Boids) 영감에 기반한 세 가지 항의 조합**으로 정의한다.

---

### 엔트로피와 보정 관점

본 연구는 어텐션을 분석하기 위해 두 가지 진단 지표를 사용한다:

* **어텐션 엔트로피**(attention entropy): 쿼리 (i)에 대한 엔트로피는
  [
  H_i=-!\sum_{j}\tilde{A}*{ij}\log\tilde{A}*{ij}.
  ]
  이는 분포가 얼마나 집중되었는지 나타내며, 값이 낮을수록 특정 이웃에 초점이 맞춰졌음을 의미한다. 다만 지나치게 낮으면 붕괴(collapsed attention)를 시사할 수 있다.

* **기대 보정 오차(Expected Calibration Error, ECE)**: 예측 확률과 실제 정확도의 차이를 나타내는 지표로, 신뢰도(confidence) 구간 ({B_m})에 대해 다음과 같이 정의된다:
  [
  \mathrm{ECE}=\sum_{m=1}^M \frac{|B_m|}{n},\big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\big|.
  ]

이 지표들은 ASCender가 단순 정확도 개선을 넘어 **해석 가능성**과 **보정 능력**을 강화하는지를 평가하는 데 사용된다.

---

### 계산적 고려사항

ASCender의 편향은 **로짓 단계에서 가산(additive)** 되며, 헤드 단위로 계산된다. 이는 정확한 자기어텐션 연산을 그대로 유지하고, FlashAttention과 같은 커널 최적화와도 호환된다. 즉, (B_h)를 softmax 이전 단계에서 단순히 추가하기만 하면 되므로 **구현의 단순성**을 유지한다.

---

## 4.2 Boids 모델의 기본 원리 (Boids Model Fundamentals)

### 보이드 모델의 세 가지 규칙

Craig Reynolds가 제안한 Boids 모델은 **개별 에이전트가 단순한 지역 규칙(local rules)만 따르더라도 전체적으로 복잡한 군집 행동(flocking behavior)이 출현**할 수 있음을 보여주었다. 핵심 규칙은 다음과 같다:

1. **정렬(Alignment):** 주변 이웃들의 평균 **방향(heading)**과 일치하려는 경향.
2. **분리(Separation):** 이웃과 과도하게 가까워지는 것을 피하려는 경향(충돌 회피).
3. **응집(Cohesion):** 주변 이웃의 평균 위치로 이동하려는 경향(군집 유지).

---

### 토큰 수준으로의 해석 (Agents → Tokens)

우리는 Transformer의 토큰을 Boids 모델의 에이전트로 해석한다. 이를 위해 두 가지 기본 연산자를 정의한다:

* **이웃 연산자(neighborhood operator) (\mathcal{N}_i):** 토큰 (i)의 지역적 이웃 집합.

  * **k-NN(content):** 임베딩 공간에서 코사인 유사도로 가장 가까운 상위 (k)개.
  * **반경 기반(radius):** 시퀀스 내 거리 (|i-j|\le r)를 만족하는 토큰들.
  * **혼합(Hybrid):** 위 두 가지의 합집합 또는 교집합.

* **방향 벡터(heading vector) (u_i):** 토큰의 의미적 방향을 나타내는 벡터.

  * (u_i=\mathrm{norm}(Q_{h,i})) 또는 (u_i=\mathrm{norm}(K_{h,i}))
  * 혹은 (Q)와 (K)의 결합 벡터로 정의 가능.

이와 함께:

* **지역 중심(local centroid):**
  [
  c_i=\frac{1}{|\mathcal{N}*i|}\sum*{j\in\mathcal{N}_i} z_j,
  ]
  여기서 (z_j)는 의미적 좌표 또는 위치 좌표.

* **지역 밀도(local density):**
  [
  \rho_i = \frac{1}{|\mathcal{N}*i|}\sum*{j\in\mathcal{N}_i}\kappa(d(i,j)),
  ]
  여기서 (\kappa)는 거리 (d(i,j))에 대한 커널 함수.

---

### 바이어스 행렬로의 변환

세 가지 규칙은 로짓 공간에서 각각 다음과 같은 바이어스 항으로 구현된다:

1. **정렬 바이어스 (B^\text{align}):**
   [
   B^\text{align}_{ij} ;=; s!\left(\langle u_i, u_j\rangle\right)\cdot \mathbf{1}[j\in\mathcal{N}_i \ \text{또는}\ i\in\mathcal{N}_j],
   ]
   → 의미적으로 방향이 유사한 토큰을 강화.

2. **분리 바이어스 (B^\text{sep}):**
   [
   B^\text{sep}*{ij} ;=; -,\lambda*\text{dist},\psi!\big(d(i,j)\big);-;\lambda_\rho,g(\rho_j),
   ]
   → 과밀하거나 중복된 토큰에 대한 어텐션을 억제.

3. **응집 바이어스 (B^\text{coh}):**
   [
   B^\text{coh}_{ij} ;=; \phi!\big(\langle z_j,, c_i\rangle - \tau\big)\cdot \mathbf{1}[j\in\mathcal{N}_i],
   ]
   → 지역 중심 근처의 토큰을 선호.

---

### 정규화 및 결합

각 바이어스 행렬은 행 단위로 정규화(z-score)하여 스케일을 맞춘다. 최종적으로 ASCender의 바이어스는 다음과 같이 정의된다:

[
B_h ;=; \beta_\mathrm{A},B^\text{align}*h;+;\beta*\mathrm{S},B^\text{sep}*h;+;\beta*\mathrm{C},B^\text{coh}_h,
]

여기서 (\beta_\cdot)는 학습되거나 사전 설정되는 조절 계수다.

---

👉 이제 이어서 **5. ASCender Model**을 한국어로 번역해드릴까요? (수식 정식화 + 알고리즘 의사코드 포함)
