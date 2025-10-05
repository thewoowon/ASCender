좋아요. **Related Work**를 국제 학회 스타일로, **영어/한국어** 모두 드릴게요.
분류 체계를 먼저 세우고, 각 축에서 무엇이 해결되었고 무엇이 남았는지, 그리고 **ASCender가 어디에 위치하며 무엇을 새로 연다**까지 이어집니다.

# Related Work

## 1) Efficiency-oriented Attention (sparsity, low-rank, kernels, memory)

A large body of work improves the **computational profile** of self-attention without changing its relational semantics.
Local/block sparsity (e.g., sliding-window and block patterns), clustering/routing, global tokens, and random links reduce quadratic cost while preserving coverage guarantees. Low-rank factorization and kernelized approximations linearize attention by constraining the score matrix structure. CUDA-level optimizations (e.g., fused kernels) further cut wall-clock time and memory.
**Relevance to ASCender.** These methods optimize *how much* we compute, not *what relations* should be preferred. ASCender is orthogonal: it **reshapes relational preferences** via inductive biases (Alignment/Separation/Cohesion), and can be stacked on top of sparse/linear/flash variants.

## 2) Positional & Distance Biases

Absolute, relative, and rotary encodings inject **order and distance**; linear distance penalties (additive biases) help length generalization, and segment-aware encodings help document-scale context.
**Gap.** These are **geometry-only** priors. They specify *where* tokens are, not *how they ought to interact semantically*.
**ASCender.** Adds **content-aware** biases: neighbors with similar semantics are encouraged to **align/cohere**, while redundant/noisy neighbors are **repelled**, beyond mere distance.

## 3) Inductive Biases in Attention

Prior work introduces **task- or structure-aware** priors: syntax/graph-informed edges, algorithmic reasoning hints (e.g., stack/queue-like regularizers), and algebra-friendly bias terms that stabilize arithmetic generalization. These approaches show that **explicit structure** can improve data efficiency and out-of-distribution behavior.
**ASCender’s angle.** Rather than injecting a *fixed* known structure (e.g., parse trees), ASCender uses **emergent-dynamics priors**—a small set of continuous rules (A/S/C) that **adapt per context** and **co-evolve with learning**, bridging from local rules to global grouping.

## 4) Grouping, Clustering, and Routing with Attention

Clustered/routing Transformers learn to **partition tokens into groups** to cut cost and sometimes improve locality. Recent analysis works report **emergent clustering** in trained attention maps; vision literature studies **object grouping** and affinity-based attention consistent with human perception. Slot-style modules and object-centric learning likewise aim for **discrete entities**.
**Difference.** Many methods *observe* or *exploit* grouping; ASCender **causes** grouping through **explicit dynamics-style biases**, yielding **steerable** (and hence more interpretable) maps.

## 5) Biologically & Swarm-Inspired Learning

Swarm intelligence (e.g., Boids rules, PSO) has informed optimization, control, and multi-agent coordination. In deep nets, biologically inspired components (lateral inhibition, normalization, recurrence) are used, but **directly embedding swarm rules into attention scores** is rare.
**ASCender.** To our knowledge, ASCender is among the first to **translate Alignment/Separation/Cohesion into additive attention terms** that operate at score level, offering a principled route from **agent-based dynamics** to **token interactions**.

## 6) Interpretability of Attention

Debates around “attention as explanation” note that raw weights can be **entangled** with other factors. Follow-ups propose constraints or probes to **stabilize** interpretations.
**ASCender.** By construction, A/S/C biases are **semantically labeled** forces. Visualizations decompose total scores into **(vanilla) + (Alignment) + (Separation) + (Cohesion)**, giving **named, inspectable contributors** to each edge.

## 7) Long-Context Modeling

Long-context models combine positional schemes, recurrence/memory, and sparse patterns. They tackle **reach** and **cost**, but not necessarily **relational selectivity**.
**ASCender.** Reduces **irrelevant long-range interactions** by biasing away from redundancy (Separation) and towards salient clusters (Cohesion), complementing long-context toolkits.

### Summary of Gaps and Our Position

Existing lines either (i) **speed up** attention, (ii) encode **geometry**, (iii) inject **discrete structure**, or (iv) post-hoc **analyze** maps. ASCender instead provides a **continuous, content-aware, dynamics-inspired inductive bias** that:

* **steers** attention to form meaningful clusters (Alignment/Cohesion),
* **suppresses** redundancy/noise (Separation), and
* remains **compatible** with sparsity, long-context memory, and standard encodings.

---

# 관련 연구 (국문)

## 1) 효율화 지향 Attention (희소화, 저랭크, 커널, 메모리)

Self-Attention의 **계산 복잡도**를 낮추는 연구들입니다.
슬라이딩 윈도·블록·클러스터/라우팅·글로벌 토큰·랜덤 링크 등으로 밀도를 줄이고, 저랭크/커널 기법으로 선형화하며, 커널 융합으로 실제 시간과 메모리를 절감합니다.
**ASCender와의 관계.** 이들은 “얼마나 덜 계산할 것인가”에 집중합니다. ASCender는 “**무엇을 우선 연결할 것인가**”를 바꾸므로 **직교적**이며, 서로 결합 가능합니다.

## 2) 위치·거리 편향

절대/상대/회전형 위치 인코딩과 선형 거리 패널티는 길이 일반화와 문서 단위 컨텍스트에 효과적입니다.
**한계.** 이는 **기하학 중심**으로, 토큰 간 **의미적 상호작용 방향성**은 규정하지 않습니다.
**ASCender.** 의미 유사 이웃과 **정렬·응집**, 과잉/잡음 이웃과 **분리**를 유도하는 **내용 기반** 편향을 추가합니다.

## 3) 유도 편향(Inductive Bias)

구문/그래프 기반 엣지, 알고리즘/산술 일반화를 돕는 규제 등 **구조적 힌트**를 직접 주입하는 연구가 있습니다. 데이터 효율과 OOD 일반화가 개선됨이 보고됩니다.
**ASCender 관점.** 고정 구조(예: 구문트리) 주입과 달리, ASCender는 \*\*동역학적 연속 규칙(A/S/C)\*\*을 사용해 **상황 적응적**으로 군집과 억제를 유도합니다.

## 4) 군집화·그룹핑·라우팅

주의를 **그룹 단위**로 조직하거나, 학습된 라우팅/클러스터링을 통해 비용을 줄이는 방법들, 그리고 **자발적 군집**이 드러난다는 분석 연구가 존재합니다.
**차이점.** 기존은 군집을 *관찰/활용*하는 경향이 강한 반면, ASCender는 **명시적 편향**으로 **군집을 생성·유도**합니다.

## 5) 생물·스웜 영감

스웜 지능(Boids, PSO 등)은 최적화나 제어에서 널리 쓰였으나, **주의 점수 수준에 Boids 규칙을 직접 삽입**하는 접근은 드뭅니다.
**ASCender.** Alignment/Separation/Cohesion을 **가산 편향 항**으로 공식화하여, **에이전트 동역학 → 토큰 상호작용**의 직접적 매핑을 제시합니다.

## 6) 주의 해석 가능성

주의가 곧 설명은 아니라는 비판 이후, 안정화·분해 기법이 제안되었습니다.
**ASCender.** 주의 점수를 \*\*(기본) + (정렬) + (분리) + (응집)\*\*으로 분해해 **명명된 기여도**를 시각화할 수 있어 해석 가능성을 높입니다.

## 7) 장문 컨텍스트

긴 문맥은 위치 체계, 재귀/메모리, 희소 패턴의 결합으로 해결해왔습니다. 이는 도달성·비용 문제를 줄이지만, **관계 선택성**은 별도 과제입니다.
**ASCender.** Separation으로 **불필요 원거리 상호작용 억제**, Cohesion으로 **핵심 군집 강화**를 유도해 장문 처리와 상호 보완적입니다.

### 정리 및 위치 지정

기존 연구는 (i) **속도**, (ii) **기하학**, (iii) **고정 구조 주입**, (iv) **사후 해석**에 집중해 왔습니다. ASCender는 **연속·내용 기반·동역학적 유도 편향**을 통해

* **의미 군집을 형성**(정렬/응집),
* **중복·잡음을 억제**(분리),
* 희소·선형·플래시·장문 메모리와 **호환**되는 길을 제시합니다.

---

원하시면 다음으로 **Method (ASCender 수식 & 알고리즘)** 섹션을 이어서 작성하겠습니다.

* 점수 행렬: $S = \frac{QK^\top}{\sqrt{d}} + \beta_{\text{align}} + \beta_{\text{sep}} + \beta_{\text{coh}}$
* 각 $\beta$의 정의(내용/거리/정규화), 미분가능한 형태, 안정화(클리핑·온도), 계산 복잡도, 기존 효율 기법과의 합성 방식을 **공식/의사코드/복잡도 표**까지 포함하여 정리해 드릴게요. 이어갈까요?
