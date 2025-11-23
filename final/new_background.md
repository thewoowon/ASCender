# 🔥 **4.2 Boids 모델의 기본 원리 (수정본, ASCender 최종 수식 반영)**

## **보이드 모델의 세 가지 규칙**

Craig Reynolds가 제안한 Boids 모델은 개별 에이전트가 단순한 지역 규칙(local rules)만으로도 전체적으로 복잡한 군집 행동(flocking behavior)이 출현할 수 있음을 보여준다. Boids의 핵심 규칙은 다음과 같다.

1. **정렬(Alignment)**: 주변 이웃의 *방향(heading)*과 자신의 방향을 일치시키려는 경향.
2. **분리(Separation)**: 이웃과 지나치게 가까워지는 것을 회피하는 경향(충돌 방지).
3. **응집(Cohesion)**: 주변 이웃의 평균 위치(centroid)를 향해 이동하려는 경향.

이 세 규칙은 모두 **지역 상호작용(local interaction)**을 기반으로 하며, 전역적 제어 없이도 자연스러운 구조적 패턴을 형성한다.

---

## **토큰 수준으로의 해석 (Agents → Tokens)**

ASCender는 Transformer의 **각 토큰을 Boids의 에이전트로 해석**한다. 이를 위해 Boids의 개념을 토큰 공간(token space)에 맞게 다음과 같이 일반화한다.

### **(1) 이웃 연산자 (Neighborhood Operator)**

토큰 (i)의 이웃 집합 (\mathcal{N}(i))는 다음의 방식 중 하나로 정의된다.

* **k-NN(content-based)**
  [
  \mathcal{N}*{k}(i)=\operatorname{TopK}*{j} \big( \cos(h_i,h_j)\big)
  ]
  임베딩 공간에서 코사인 유사도가 가장 높은 (k)개 토큰을 선택.

* **반경 기반(radius-based)**
  [
  \mathcal{N}_{r}(i)={j : |i-j|\le r}
  ]
  시퀀스 또는 3D 점군에서 거리 기준으로 선택.

* **혼합(Hybrid)**
  [
  \mathcal{N}(i)=\mathcal{N}*{k}(i)\cup\mathcal{N}*{r}(i)
  ]

이웃의 정의는 Boids의 “local awareness”에 대응한다.

---

### **(2) 방향 벡터 (Semantic Heading Vector)**

Boids의 heading 개념은 토큰의 의미적 방향 벡터로 일반화된다.

[
u_i = \frac{q_i}{|q_i|}
\quad \text{또는} \quad
u'_i = \frac{k_i}{|k_i|}
]

즉, normalized Q/K 벡터가 **의미적 진행 방향(semantic heading)** 역할을 한다.

---

### **(3) 지역 중심 (Local Centroid)**

[
c_i = \frac{1}{|\mathcal{N}(i)|}
\sum_{j\in\mathcal{N}(i)} z_j
]

여기서

* (z_j)는 임베딩 좌표 또는 3D 포인트 좌표.

---

### **(4) 지역 밀도 (Local Density)**

[
\rho_i=
\sum_{j\in\mathcal{N}(i)}\kappa(d(i,j))
]

여기서

* (\kappa(\cdot))는 Gaussian kernel,
* (d(i,j))는 위치 또는 좌표 기반 거리.

---

## **바이어스 행렬로의 변환 (Boids → ASCender Bias)**

Boids의 규칙을 Transformer attention logit에서의 **additive bias** 형태로 변환하면 다음과 같다.
아래는 기존 Boids 규칙을 ASCender가 실제 사용하는 **최종 수식 기반**으로 재정식화한 것이다.

---

### **1. 정렬 바이어스 (Alignment Bias)**

Boids의 “heading alignment”는 **코사인 유사도 기반**의 정렬 편향으로 표현된다.

[
B^{\text{align}}*{ij}
=\frac{
\langle u_i,u_j\rangle
}{\tau}
=\frac{
\langle \hat{q}*{i},\hat{k}_{j}\rangle
}{\tau}
]

* 방향이 유사할수록 높은 양의 bias → attention 강화
* ASCender alignment의 **최종 정확한 형태와 일치**

---

### **2. 분리 바이어스 (Separation Bias)**

Boids의 “collision avoidance”는 **Gaussian repulsion**으로 구현된다.

[
B^{\text{sep}}_{ij}
=\exp\left(-\frac{|i-j|^2}{2\sigma_S^2}\right)
]

ASCender에서는 분리 규칙은 **항상 음수(반발)**로 작용하므로:

[
-,w_S,B^{\text{sep}}_{ij}
]

* 가까울수록 강한 음수 값 → 과밀 영역의 attention 억제
* ASCender separation의 **최종 구현과 동일**

---

### **3. 응집 바이어스 (Cohesion Bias)**

Boids의 “moving toward centroid”는 **Gaussian attraction**으로 표현된다.

[
B^{\text{coh}}_{ij}
=\exp\left(-\frac{|i-j|^2}{2\sigma_C^2}\right)
]

ASCender에서는 응집은 항상 **양의 bias**로 작용한다:

[
+,w_C,B^{\text{coh}}_{ij}
]

* 근접 토큰 강화, local grouping 형성
* Cohesion의 ASCender 최종 모델 정의와 정확히 일치

---

## **정규화 및 결합 (Normalization and Combination)**

ASCender의 최종 bias는 Boids 기반 세 규칙을 가중합으로 결합하여 구성된다.

[
B_h(i,j)
========

\gamma_h,\sigma_h,
\mathrm{clamp}
\left[
w_A B^{\text{align}}_{ij}

* w_C B^{\text{coh}}_{ij}

- w_S B^{\text{sep}}_{ij}
  \right]
  ]

여기서

* (w_{\cdot}): alignment/separation/cohesion weight
* (\gamma_h): head-specific scale
* (\sigma_h): learnable gate
* clamp((\cdot)): 안정성 보장을 위한 clipping
* band-pass mask 및 causal mask는 최종 단계에서 적용됨

정규화(z-score)는 선택적이며, 전체 구조의 보존을 위해 실험에서는 비활성화하였다.

---

# 🔥 요약 (연구자가 reviewer에게 보여줄 수준)

* Alignment = **정규화된 Q/K 코사인 유사도**
* Separation = **좁은 Gaussian repulsion**
* Cohesion = **넓은 Gaussian attraction**
* ASCender bias = **learnable scaled-gated weighted sum**

---

필요하다면:
**“5. ASCender Model” 전체 한국어 정리**,
**수식 번호 매기기**,
**LaTeX 버전**,
**numpy/pytorch pseudo-code 버전**까지 이어서 구성해줄게.
