# ASCender

ASCender

Craig. W. Reynolds

Flocks, Herds, and Schools: A Distributed Behavioral Model

Computer Graphics, 21(4), July 1987, pp. 25-34.
http://www.red3d.com/cwr/papers/1987/boids.html

Boids is an artificial life program, developed by Craig Reynolds in 1986, which simulates the flocking behavior of birds. The name "Boids" is a play on the word "birds" and refers to a bird-like object. Each boid follows three simple rules:

1. Separation: steer to avoid crowding local flockmates

2. Alignment: steer towards the average heading of local flockmates

3. Cohesion: steer to move toward the average position of local flockmates

By following these rules, boids can create complex and realistic flocking behaviors. The model has been widely used in computer graphics, animation, and artificial intelligence to simulate the behavior of groups of animals.

ASCender는 Craig Reynolds의 Boids 모델에 기반해 Transformer 아키텍처를 재해석하려는 아주 흥미로운 시도.

그 구조적 흐름과 전제, 그리고 ASCender의 핵심 구성요소들을 체계적으로 정리합니다.

## 1. 영감의 출발점: Craig Reynolds의 Boids 모델 

### 📘 논문: Flocks, Herds, and Schools: A Distributed Behavioral Model (1987) 

Craig Reynolds는 개별 에이전트가 단순한 지역 규칙만을 따름으로써 전체적으로 군집 형태를 만들어내는 Boids 알고리즘을 제안했다. 이 모델은 군중, 떼, 무리를 시뮬레이션하는데 탁월했고, 주요한 세 가지 규칙으로 구성된다:

1. Separation (분리): 너무 가까운 이웃으로부터 떨어지려는 경향 (충돌 방지)

2. Alignment (정렬): 주변 보이드와 방향을 일치시키려는 경향 (속도/방향의 동기화)

3. Cohesion (응집): 이웃을 향해 뭉치려는 경향 (무리 유지)

## 2. ASCender의 출발점 

Transformer는 모든 토큰 간의 global attention을 계산하는 구조지만, 이는: 

* 비효율적인 연산 복잡도 O(n^2) 
* 각 토큰의 역할 차별화 부족 
* Position 정보와 의미 기반 결속이 약함 이라는 한계를 가짐. 

이에 우리는 Boids 모델의 로컬 상호작용 기반 군집 형성 원리를 Transformer Attention에 통합하는 모델, ASCender를 제안함.


## 3. ASCender의 핵심 전제 

### 🎯 목표 

* 기존 Self-Attention을 보완하여 더 구조화된 attention 패턴 유도 
* 각 토큰이 구조적 상호작용(local)을 기반으로 정보를 수용하도록 설계 
* Alignment, Separation, Cohesion을 attention bias로 수식화하여 적용

## 4. ASCender 구조 개요 

### 🌐 Base: Transformer Self-Attention 

Self-attention의 기본 구조는 유지하되, **각 토큰 간 관계에 Boids-inspired bias를 추가**하여 attention score를 조정함. #### 기존 Attention Score 계산:

math
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V

#### ASCender Attention 수정:

math
\text{ASCenderAttention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + \beta \cdot B \right) V

여기서 B는 아래의 세 요소를 합산한 Boid-inspired Attention Bias Matrix이고, β는 조절 계수임.


순서

1. 실제 Self-Attention 식을 작성
2. ASCender Attention Alignment
3. ASCender Attention Separation
4. ASCender Attention Cohesion
5. ASCender Attention Bias Matrix

## 7. 특징 및 기대 효과
| 구성 요소 | 역할 | 기대 효과 | 
| ---------- | ----------------------- | ------------------------------ | 
| Alignment | 방향 정렬 기반 집중 강화 | 의미/문맥 유사성 기반 이해 향상 | 
| Separation | 과밀 토큰에 대한 주의 억제 | 중복 억제, 정보 다양성 유지 | 
| Cohesion | 의미 기반 뭉침 유도 | 문장 내 논리적 흐름 강화 | 
| 구조화된 Bias | Attention 스코어에 구조 정보 주입 | Emergent-like attention map 형성 |


"각성한 Attention은 무리 속에서 길을 찾는다." — ASCender.

thesis structure

1. Abstract
2. Introduction
3. Related Work
4. Background: Transformer and Self-Attention
5. Boids Model Overview
6. ASCender Model
   1. Base Transformer Attention
   2. Alignment Component
   3. Separation Component
   4. Cohesion Component
   5. Combined Attention Bias
7. Experiments
   1. Datasets and Setup
   2. Baseline Comparisons
   3. Ablation Studies
8. Results and Analysis
   1. Quantitative Results
   2. Qualitative Analysis
   3. Attention Visualization
9. Discussion
   1. Insights from Boids Integration
   2. Limitations and Future Work
10. Conclusion
11. References

