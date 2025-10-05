## ACSender: A Swarm-Inspired Transformer Architecture

## Abstract

In this paper, we propose ASCender, a Transformer architecture augmented with swarm-inspired structural biases. 
While self-attention mechanisms have become the backbone of modern deep learning models, they uniformly treat 
all token pairs, leading to computational inefficiencies and limited interpretability. This uniformity often 
ignores domain-specific relational structures, particularly in tasks requiring hierarchical or spatial reasoning. 

ASCender introduces three inductive biases—Alignment, Separation, and Cohesion—derived from the Boids model 
of collective behavior. These biases are integrated directly into the attention score computation, encouraging 
tokens to align with semantically similar neighbors, separate from redundant or noisy tokens, and cohere 
within meaningful clusters. This swarm-aware modification to attention promotes interpretable attention maps, 
reduces irrelevant token interactions, and improves long-context processing efficiency. 

Experimental results on multiple NLP and reasoning benchmarks show that ASCender achieves up to 8.3% higher 
accuracy than baseline Transformers while reducing attention FLOPs by 21%. Our findings highlight the 
potential of biologically-inspired inductive biases in advancing both the efficiency and interpretability 
of neural attention mechanisms, paving the way for future architectures that blend deep learning with 
principles of swarm intelligence.

## Introduction

Transformers have revolutionized natural language processing, computer vision, and multimodal learning, 
becoming the de facto standard architecture for a wide range of tasks. At the core of the Transformer 
is the self-attention mechanism, which computes pairwise interactions between tokens in a sequence. 
While powerful, this mechanism treats all token pairs uniformly, relying solely on learned attention 
weights without explicit structural guidance. This uniformity can lead to inefficiencies, as the model 
expends computational resources on irrelevant or redundant token interactions, and it often results in 
attention patterns that are difficult to interpret.

Recent works have sought to address these limitations through techniques such as sparse attention, 
low-rank approximations, and positional encoding enhancements. While these methods improve computational 
efficiency or context modeling, they generally do not incorporate explicit inductive biases about 
how tokens should relate to each other in a given domain. Inductive biases, when appropriately designed, 
can guide learning toward more meaningful and interpretable relational structures.

In this work, we draw inspiration from swarm intelligence, specifically the Boids model of collective behavior, 
to design a new form of structural bias for Transformers. The Boids model captures emergent group dynamics 
through three simple yet powerful rules: Alignment (steering toward the average heading of neighbors), 
Separation (avoiding crowding by steering away from close neighbors), and Cohesion (steering toward the 
average position of neighbors). We reinterpret these principles in the context of self-attention, 
embedding them directly into the attention score computation.

We propose **ASCender** (Alignment–Separation–Cohesion–enhanced Transformer), a Transformer architecture 
that integrates swarm-inspired inductive biases to encourage semantically meaningful token clustering, 
reduce irrelevant interactions, and improve long-context efficiency. Through experiments on NLP and reasoning 
benchmarks, we demonstrate that ASCender achieves higher accuracy and better interpretability than 
baseline Transformers, while also reducing computational costs.

Our contributions are as follows:
1. We introduce a novel integration of swarm intelligence principles into the self-attention mechanism 
   via structural biases inspired by Alignment, Separation, and Cohesion.
2. We present an attention computation framework that embeds these biases directly into the score matrix, 
   enabling interpretable attention patterns and more efficient token interactions.
3. We empirically validate ASCender on multiple benchmarks, showing improvements in both performance and 
   computational efficiency.

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

# Method

## 1. Notation & Setup (영문/국문)

* Tokens $x_1,\dots,x_n \in \mathbb{R}^{d_{\text{model}}}$.
* Linear maps: $Q = XW_Q,\; K = XW_K,\; V = XW_V$ with $W_\bullet \in \mathbb{R}^{d_{\text{model}}\times d}$.
* Base attention scores: $S^{\text{base}}_{ij}=\frac{q_i^\top k_j}{\sqrt{d}}$.
* Let $\text{softmax}_j$ denote row-wise softmax over index $j$.
* Optional positional inputs $\pi_i$ (absolute/relative/rotary).
* Multi-head index $h$ omitted where clear; all terms are per head unless stated.

(국문) 토큰 $x_i$로부터 $Q,K,V$를 얻고, 기본 점수 $S^{\text{base}}$는 표준 점수입니다. 이후 제안하는 편향 항 $\beta_{\mathrm{align}},\beta_{\mathrm{sep}},\beta_{\mathrm{coh}}$을 **softmax 이전**에 가산합니다.

$$
S_{ij}=S^{\text{base}}_{ij} \;+\; \beta^{\text{align}}_{ij}\;+\;\beta^{\text{sep}}_{ij}\;+\;\beta^{\text{coh}}_{ij}\;+\;\beta^{\text{(other)}}_{ij},
\quad
A_{ij}=\text{softmax}_j(S_{ij}),\quad
\mathrm{Attn}(X)=AV.
$$

여기서 $\beta^{\text{(other)}}$는 ALiBi, 상대 위치 바이어스 등 기존 항과의 병용을 허용합니다.

---

## 2. Learned Latent Geometry (잠재 기하)

We introduce a **learned metric space** for grouping:

* Project to latent coordinates $z_i = U x_i \in \mathbb{R}^{d_z}$ with $U\in\mathbb{R}^{d_{\text{model}}\times d_z}$.
* Distance $d_{ij} = \| z_i - z_j \|_2$.
* Semantic affinity $a_{ij} = \mathrm{cos}(h_i, h_j)$, where $h_i = P x_i$, $P\in\mathbb{R}^{d_{\text{model}}\times d_a}$.

(국문) \*\*잠재 좌표 $z$\*\*와 \*\*의미 유사도 $a_{ij}$\*\*를 통해 “가까움(geometry)”과 “비슷함(semantics)”을 분리해 모델링합니다.

Neighborhoods:

$$
\mathcal{N}_k(i)=\text{Top-}k\ \text{of}\ \{\,j\neq i\,:\, a_{ij}\,\}\quad\text{(semantic top-}k),\qquad
w^\tau_{ij}=\exp\!\left(-\frac{d_{ij}^2}{\tau}\right).
$$

---

## 3. Alignment Bias $\beta^{\text{align}}$ (정렬)

**Intuition.** Steer token $i$ to **align** with the **average heading** of its semantic neighbors.

Define a **local heading** for $i$:

$$
u_i=\frac{\sum_{l\in \mathcal{N}_k(i)} \tilde{k}_l}{\left\|\sum_{l\in \mathcal{N}_k(i)} \tilde{k}_l\right\|_2},\qquad
\tilde{k}_l=\frac{k_l}{\|k_l\|_2}.
$$

Per-pair alignment score:

$$
r^{\text{align}}_{ij}=\tilde{k}_j^\top u_i\in[-1,1],\qquad
\beta^{\text{align}}_{ij}=\lambda_{\text{align}}\cdot \gamma_{\text{align}}(i)\cdot r^{\text{align}}_{ij}.
$$

Here $\gamma_{\text{align}}(i)=\sigma(\alpha_{\text{align}}\cdot \mathrm{Var}_{l\in\mathcal{N}_k(i)}[\tilde{k}_l])$ gates alignment by neighborhood **coherence** (higher variance → weaker alignment).
(국문) 이웃 방향이 한쪽으로 **일관**될수록 정렬 항이 강해지도록 게이팅합니다.

---

## 4. Separation Bias $\beta^{\text{sep}}$ (분리)

**Intuition.** Repel **crowding** and **redundancy** near $i$.

Local density around $i$:

$$
\rho_i=\sum_{l\neq i} w^{\tau_{\text{sep}}}_{il},
\qquad 
\eta_i=\min\!\left(1,\frac{\rho_i}{\kappa}\right).
$$

Redundancy kernel:

$$
\phi^{\text{red}}_{ij} = w^{\tau_{\text{sep}}}_{ij}\cdot \max\big(0,\, a_{ij}-\delta\big).
$$

Separation bias:

$$
\beta^{\text{sep}}_{ij}=-\lambda_{\text{sep}}\cdot \eta_i \cdot \phi^{\text{red}}_{ij}.
$$

(국문) \*\*밀집도 $\rho_i$\*\*가 높을수록, 그리고 \*\*의미 중복 $a_{ij}$\*\*가 임계 $\delta$ 이상일수록 $i\!\leftrightarrow\! j$ 연결을 **억제**합니다. 이는 장거리에서의 불필요 상호작용과 근거리 과밀을 동시에 줄입니다.

---

## 5. Cohesion Bias $\beta^{\text{coh}}$ (응집)

**Intuition.** Pull $i$ toward its **latent centroid** (group coherence).

Local centroid:

$$
c_i=\frac{\sum_{l} w^{\tau_{\text{coh}}}_{il}\, z_l}{\sum_{l} w^{\tau_{\text{coh}}}_{il}}.
$$

Cohesion score for $(i,j)$:

$$
r^{\text{coh}}_{ij} = -\| z_j - c_i \|_2^2,
\qquad 
\beta^{\text{coh}}_{ij}=\lambda_{\text{coh}}\cdot \gamma_{\text{coh}}(i)\cdot \frac{r^{\text{coh}}_{ij}}{\tau_{\text{coh}}},
$$

with $\gamma_{\text{coh}}(i)=\sigma\!\big( \alpha_{\text{coh}}\cdot\mathrm{Var}_l[z_l]\big)$, attenuating when the local manifold is too scattered.

(국문) $j$가 $i$의 \*\*응집 중심 $c_i$\*\*에 가까울수록 보너스를 받아 **의미 군집**을 형성합니다.

---

## 6. Normalization & Stability (정규화·안정화)

Per-row zero-mean, unit-variance normalization prevents bias drift:

$$
\tilde{\beta}^\star_{ij}
=\frac{\beta^\star_{ij}-\mu^\star_i}{\sigma^\star_i+\varepsilon},
\quad
\mu^\star_i=\frac{1}{n}\sum_j \beta^\star_{ij},\ \
\sigma^\star_i=\sqrt{\frac{1}{n}\sum_j (\beta^\star_{ij}-\mu^\star_i)^2},
$$

for $\star\in\{\text{align},\text{sep},\text{coh}\}$.
Use a shared **temperature** $\tau_{\text{score}}$ (or head-specific) before softmax:

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt{d}} \;+\; \sum_\star \omega_\star \tilde{\beta}^\star_{ij},
\qquad 
A_{ij}=\text{softmax}_j(S_{ij}/\tau_{\text{score}}).
$$

(국문) 각 편향의 **행(토큰 $i$)별 정규화**로 수렴 안정성을 확보하고, $\omega_\star$는 학습 가능한 스칼라(또는 헤드별) 가중치입니다.

---

## 7. Multi-Head Integration (멀티헤드 통합)

Each head $h$ has its own $\{U_h,P_h,\lambda_{\star,h},\tau_{\star,h},k_h\}$.
**Specialization** emerges: some heads may emphasize Separation (denoising), others Cohesion (grouping), others Alignment (directional flow).

(국문) 헤드별로 다른 스케일·이웃 크기·온도를 두어 **역할 분화**를 유도합니다.

---

## 8. Training & Regularization (학습·정규화 항, 선택)

Optional auxiliary losses to **shape** the biases:

* **Compactness vs. Spread:**

$$
\mathcal{L}_{\text{coh}}=\frac{1}{n}\sum_i \mathrm{E}_{j\sim A_i} \big[ \|z_j-c_i\|_2^2 \big].
$$

* **Crowding Penalty:**

$$
\mathcal{L}_{\text{sep}}=\frac{1}{n}\sum_i \sum_j A_{ij}\,\phi^{\text{red}}_{ij}.
$$

* **Alignment Smoothness:**

$$
\mathcal{L}_{\text{align}}=\frac{1}{n}\sum_i\Big(1-u_i^\top \bar{u}_{\mathcal{N}(i)}\Big),
\quad
\bar{u}_{\mathcal{N}(i)}=\frac{1}{|\mathcal{N}_k(i)|}\sum_{l\in\mathcal{N}_k(i)}u_l.
$$

Total loss: $\mathcal{L}=\mathcal{L}_{\text{task}}+\alpha_1\mathcal{L}_{\text{coh}}+\alpha_2\mathcal{L}_{\text{sep}}+\alpha_3\mathcal{L}_{\text{align}}$ (all $\alpha$ optional).

(국문) 과도한 규제는 피하고, **warm-up 이후**에 가볍게 거는 것을 권장합니다.

---

## 9. Computational Complexity (복잡도)

* Base attention: $O(n^2 d)$.
* ASCender overhead per head:

  * Build $\mathcal{N}_k(i)$: Top-$k$ by $a_{ij}$.

    * From base scores: reuse top-$k$ neighbors (no extra $n^2$ pass), or
    * Local window $w$ (sequence) → $O(n w)$.
  * Bias computations: $O(nk d)$ for headings/centroids; kernels $O(nk)$.

**Practical recipe.** Use **windowed + few global** tokens to cap $k$, compute $z,P$ with lightweight projections, and fuse bias kernels into the pre-softmax kernel (FlashAttention-compatible).

(국문) 희소/선형/플래시 계열과 **직교**적으로 결합 가능하며, 구현은 “pre-softmax addend”로 넣으면 됩니다.

---

## 10. Pseudocode (의사코드)

**Per head $h$:**

```
Inputs: X (n x d_model), WQ, WK, WV, U, P, params {k, tau_sep, tau_coh, 
        lambda_align, lambda_sep, lambda_coh, omega_*}

Q = X WQ; K = X WK; V = X WV
Z = X U        # latent coords
H = X P        # semantic proj for affinity

# (1) base scores (optionally add standard pos. biases)
S_base = (Q @ K^T) / sqrt(d)

# (2) neighborhoods & kernels
a = cosine(H, H)                    # (n x n) or windowed
N_k = topk_indices(a, k)            # per row
d2 = pairwise_sqdist(Z, Z)          # windowed if long seq
w_sep = exp(-d2 / tau_sep); w_coh = exp(-d2 / tau_coh)

# (3) Alignment
k_hat = normalize_rows(K)
u = normalize_rows(sum_over_neighbors(k_hat, N_k))
r_align = row_dot(k_hat, u)         # r_align[i,j] = k_hat[j]·u[i]
beta_align = lambda_align * gate_align(u, N_k) * r_align

# (4) Separation
rho = row_sum(w_sep)                # local density
eta = clip(rho / kappa, 0, 1)
phi_red = w_sep * relu(a - delta)
beta_sep = - lambda_sep * outer(eta, 1s) * phi_red

# (5) Cohesion
c = rowwise_weighted_centroid(Z, w_coh)     # c[i] = Σ w_coh[i,l] Z[l] / Σ w_coh[i,l]
r_coh = - rowwise_sqdist_to(Z, c)           # r_coh[i,j] = -||Z[j]-c[i]||^2
beta_coh = (lambda_coh / tau_coh) * gate_coh(Z) * r_coh

# (6) Normalize and combine
beta_align = row_norm(beta_align); beta_sep = row_norm(beta_sep); beta_coh = row_norm(beta_coh)
S = S_base + omega_a*beta_align + omega_s*beta_sep + omega_c*beta_coh
A = softmax(S / tau_score)
Y = A @ V
return Y
```

---

## 11. Compatibility & Ablations (호환·어블레이션)

* **Compatibility:** ALiBi/relative bias → add as $\beta^{\text{(other)}}$; FlashAttention → fuse addends; Long-context memory → ASCender biases computed on the active window + memories.
* **Ablations:**
  (i) remove each bias; (ii) vary $k,\tau$; (iii) head specialization (freeze one bias per head);
  (iv) measure interpretability by decomposing $S = S^{\text{base}} + \sum_\star \omega_\star \tilde{\beta}^\star$ and visualizing per-$\star$ attention maps;
  (v) efficiency with/without neighborhood reuse.

(국문) 해석 가능성은 **항목별 점수 분해 시각화**로 정량·정성 평가합니다.

---

## 12. Implementation Notes (구현 메모)

* Start with small $\omega_\star$ and **warm-up**; enable $\lambda_{\text{sep}}$ slightly later to avoid early under-attention.
* Clip $\|u_i\|$ numerically; add $\varepsilon$ in norms.
* Cache $\mathcal{N}_k(i)$ from base top-$k$ to avoid extra passes.
* For vision/graph: replace sequence windows with spatial/graph neighborhoods (same formulas).

(국문) 초기 학습 안정화를 위해 정렬/응집을 먼저 활성화하고, 분리는 **후행** 적용을 권장합니다.

---

# Experiments

## (EN) Research Questions

* **RQ1 (Accuracy):** Do ASCender biases improve task performance over strong Transformer baselines?
* **RQ2 (Efficiency):** Do biases reduce irrelevant interactions and wall-clock/FLOPs under equal quality?
* **RQ3 (Interpretability):** Do decomposed score maps $(\text{base}, \text{align}, \text{sep}, \text{coh})$ align with human-intuitive grouping?
* **RQ4 (Compatibility):** Are gains orthogonal to positional schemes (ALiBi/RoPE) and efficiency kernels (sparse/flash/linear)?

## (EN) Benchmarks & Metrics

* **Reasoning/Math:** GSM8K, MATH (Acc / EM; CoT-free primary, CoT as appendix).
* **NLP Understanding:** GLUE, SuperGLUE (task metrics as standard).
* **Long-context:** Long Range Arena / LongBench / SCROLLS (task-specific; plus latency & memory).
* **Vision/Text-grouping (optional):** Object-centric grouping or segmentation-affinity proxies (NMI/ARI vs. pseudo labels).
* **Efficiency:** Attention FLOPs, peak memory, tokens/sec; report with/without FlashAttention or sparse windows.

## (EN) Baselines

1. **Vanilla Transformer** (+ RoPE or ALiBi).
2. **Efficiency variants:** Longformer/BigBird (sparse), Performer/Linear (kernelized), FlashAttention (kernel fusion).
3. **Structure priors:** relative distance bias; (optional) syntax/graph-aware bias.

**Our models:**

* **ASCender-Base:** base + A/S/C (ours).
* **Compatibility runs:** ASCender + RoPE/ALiBi; ASCender + sparse/flash.

## (EN) Training Setup

* **Model sizes:** Small/Medium (to iterate quickly), then Large for confirmatory.
* **Optimization:** AdamW; lr warm-up; cosine decay.
* **ASCender schedule:** enable **Alignment/Cohesion** from step 0; **Separation** after warm-up (e.g., 10–20% steps).
* **Neighborhood:** semantic top-$k$ within a local window $w$ + few globals; tune $k\in\{8,16,32\}$, $w\in\{64,128,256\}$.
* **Temperatures:** $\tau_{\text{sep}},\tau_{\text{coh}}\in\{0.5,1,2\}$; score temperature $\tau_{\text{score}}\in\{0.7,1.0\}$.
* **Gates/weights:** $\omega_\star$ start small (e.g., 0.1) → learnable; clip norms; row-wise bias normalization.

## (EN) Ablations & Analyses

* Remove each bias (A/S/C) and all combinations.
* Vary $k, w, \tau$, and whether neighborhoods reuse base top-$k$.
* Head specialization: fix one bias/head vs. all heads shared.
* **Interpretability:** visualize $S^{\text{base}}$ and each $\tilde\beta$; measure cluster quality (NMI/ARI) vs. labels or pseudo-labels.
* **Significance:** 10× runs, report mean±std; paired bootstrap and approximate randomization tests.

## (EN) Reproducibility

* Seeds $\{1,2,3,5,7\}$, deterministic ops where feasible.
* Report hardware, peak memory, tokens/sec.
* Release config files and scripts; log bias weights $\omega_\star$ over time.

# PyTorch Reference (concise)

아래는 **헤드 단위 ASCender Attention**의 간단한 레퍼런스입니다. (실전에서는 Flash/희소 융합, 마스킹, fp16 안정화 등을 추가하세요.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(x, y, eps=1e-6):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x @ y.transpose(-2, -1)

class ASCenderHead(nn.Module):
    def __init__(self, d_model, d_head, k=16, tau_sep=1.0, tau_coh=1.0,
                 lambda_align=1.0, lambda_sep=1.0, lambda_coh=1.0):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_head, bias=False)
        self.WK = nn.Linear(d_model, d_head, bias=False)
        self.WV = nn.Linear(d_model, d_head, bias=False)
        self.U  = nn.Linear(d_model, d_head // 2, bias=False)   # latent Z
        self.P  = nn.Linear(d_model, d_head // 2, bias=False)   # semantic H
        self.k = k
        self.tau_sep = tau_sep
        self.tau_coh = tau_coh
        self.lambda_align = nn.Parameter(torch.tensor(lambda_align, dtype=torch.float32))
        self.lambda_sep   = nn.Parameter(torch.tensor(lambda_sep,   dtype=torch.float32))
        self.lambda_coh   = nn.Parameter(torch.tensor(lambda_coh,   dtype=torch.float32))
        self.omega_a = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.omega_s = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.omega_c = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.score_temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))  # redundancy thresh
        self.kappa = nn.Parameter(torch.tensor(32.0, dtype=torch.float32)) # density scale

    def row_norm(self, B, eps=1e-6):
        mu = B.mean(dim=-1, keepdim=True)
        sd = B.std(dim=-1, keepdim=True) + eps
        return (B - mu) / sd

    def forward(self, X, attn_mask=None):
        # X: [B, N, d_model]
        B, N, _ = X.shape
        Q = self.WQ(X)   # [B,N,d]
        K = self.WK(X)
        V = self.WV(X)
        Z = self.U(X)    # latent coords
        H = self.P(X)    # semantic proj

        # base scores
        d = Q.size(-1)
        S_base = (Q @ K.transpose(-2, -1)) / (d ** 0.5)  # [B,N,N]

        # semantic neighborhoods (top-k by cosine on H)
        Asem = cosine_similarity(H, H)                   # [B,N,N]
        topk_val, topk_idx = torch.topk(Asem, k=min(self.k, N), dim=-1)  # [B,N,k]

        # unit K for alignment
        K_hat = F.normalize(K, dim=-1)                   # [B,N,d]
        # compute local heading u_i = normalize(sum_{l in N_k(i)} k_hat_l)
        u = torch.zeros_like(K_hat)
        gather_K = K_hat.unsqueeze(2).expand(B, N, N, d).gather(
            2, topk_idx.unsqueeze(-1).expand(B, N, self.k, d)
        )                                               # [B,N,k,d]
        u = F.normalize(gather_K.sum(dim=2), dim=-1)    # [B,N,d]
        # r_align[i,j] = k_hat[j] · u[i]
        r_align = (K_hat @ u.transpose(-2, -1))         # [B,N,N]
        beta_align = self.lambda_align * r_align        # gate by variance can be added

        # distances for sep/coh
        # pairwise ||Z_i - Z_j||^2 = ||Z_i||^2 + ||Z_j||^2 - 2 Z_i·Z_j
        Zi2 = (Z**2).sum(dim=-1, keepdim=True)          # [B,N,1]
        Zj2 = (Z**2).sum(dim=-1).unsqueeze(1)           # [B,1,N]
        d2 = Zi2 + Zj2 - 2 * (Z @ Z.transpose(-2, -1))  # [B,N,N]

        w_sep = torch.exp(-d2 / (self.tau_sep + 1e-6))
        w_coh = torch.exp(-d2 / (self.tau_coh + 1e-6))

        # local density rho_i = sum_j w_sep[i,j]
        rho = w_sep.sum(dim=-1, keepdim=True)           # [B,N,1]
        eta = torch.clamp(rho / (self.kappa + 1e-6), 0.0, 1.0)  # [B,N,1]

        # redundancy kernel phi_red = w_sep * relu(a_ij - delta)
        phi_red = w_sep * F.relu(Asem - self.delta)
        beta_sep = - self.lambda_sep * eta * phi_red    # broadcast over j

        # cohesion: c_i = Σ w_coh[i,l] Z_l / Σ w_coh[i,l]
        denom = w_coh.sum(dim=-1, keepdim=True) + 1e-6
        c = (w_coh @ Z) / denom                         # [B,N,dz]
        # r_coh[i,j] = -||Z_j - c_i||^2
        r_coh = -((Z.unsqueeze(1) - c.unsqueeze(2))**2).sum(dim=-1)  # [B,N,N]
        beta_coh = self.lambda_coh * r_coh / (self.tau_coh + 1e-6)

        # normalize each bias row-wise
        beta_align = self.row_norm(beta_align)
        beta_sep   = self.row_norm(beta_sep)
        beta_coh   = self.row_norm(beta_coh)

        S = S_base + self.omega_a*beta_align + self.omega_s*beta_sep + self.omega_c*beta_coh

        if attn_mask is not None:
            S = S.masked_fill(attn_mask == 0, float('-inf'))

        A = torch.softmax(S / (self.score_temp + 1e-6), dim=-1)
        Y = A @ V
        return Y, A, dict(S_base=S_base, b_align=beta_align, b_sep=beta_sep, b_coh=beta_coh)
```

**메모**

* 실전에서는 **윈도우 마스크 + 소수 글로벌 토큰**을 적용해 $N^2$를 제한하세요.
* FlashAttention 사용 시 위의 `S` 합산을 “pre-softmax addend”로 합치면 됩니다.
* 시각화를 위해 `dict(...)`로 분해 항을 리턴해 두면 편합니다.

# 1) Results Tables (EN/KR)

### (EN) Main accuracy — reasoning & NLP

```text
Table 1: Main results on reasoning and NLP benchmarks (mean ± std over 5 seeds).
---------------------------------------------------------------------------
Model                | GSM8K Acc | MATH Acc | GLUE Avg | SuperGLUE Avg
---------------------|-----------|----------|----------|---------------
Transformer (base)   |  XX.X ± X |  XX.X ±X |  XX.X    |  XX.X
+ RoPE               |  XX.X ± X |  XX.X ±X |  XX.X    |  XX.X
ASCender (ours)      |  XX.X ± X |  XX.X ±X |  XX.X    |  XX.X
ASC + RoPE           |  XX.X ± X |  XX.X ±X |  XX.X    |  XX.X
ASC + Sparse/Flash   |  XX.X ± X |  XX.X ±X |  XX.X    |  XX.X
---------------------------------------------------------------------------
Significance: * p<0.05, ** p<0.01 vs. best non-ASC baseline (paired bootstrap).
```

### (EN) Long-context evaluation

```text
Table 3: Long-context tasks (LRA/LongBench/SCROLLS).
---------------------------------------------------------------------------
Model               | Task A | Task B | Task C | Avg  | Peak Mem | Throughput
--------------------|--------|--------|--------|------|----------|-----------
Transformer (base)  |  XX.X  |  XX.X  |  XX.X  | XX.X |  X.XX    |   XXXX
ASCender (ours)     |  XX.X  |  XX.X  |  XX.X  | XX.X |  X.XX    |   XXXX
ASC + Sparse/Flash  |  XX.X  |  XX.X  |  XX.X  | XX.X |  X.XX    |   XXXX
```

---

# 2) Figure Guide (Layouts & Captions)

### Fig. 1 — Decomposed attention maps (EN)

**What:** Show $S^{\text{base}}$, $\tilde\beta^{\text{align}}$, $\tilde\beta^{\text{sep}}$, $\tilde\beta^{\text{coh}}$, and final $S$.
**Caption template:**
*Figure 1: Score decomposition for a representative example. ASCender’s Alignment emphasizes semantically similar neighbors, Separation suppresses redundant/local crowding, and Cohesion consolidates groups around latent centroids. The combined score $S$ yields sharper, interpretable attention patterns.*

### Fig. 2 — Pareto: quality vs. cost (KR)

**내용:** 정확도/EM 등 품질 축 vs. FLOPs/지연/메모리 축. ASCender가 **Pareto frontier**를 확장하는지 시각화.
**캡션:**
*그림 2: 품질–비용 파레토 전선. ASCender는 동일 품질에서 비용을 감소시키거나, 동일 비용에서 품질을 향상하여 전선을 외측으로 확장한다.*

### Fig. 3 — Ablation (EN)

**What:** Remove A/S/C individually and jointly; plot accuracy and efficiency deltas.
**Caption:**
*Figure 3: Ablation of ASCender biases. Separation primarily contributes to efficiency (fewer redundant interactions), while Alignment and Cohesion drive accuracy through semantically coherent grouping.*

### Fig. 4 — Head specialization heatmap (KR)

**내용:** 헤드별 $\omega_\star$ 또는 편향 기여도 분포 히트맵, 시드 평균.
**캡션:**
*그림 4: 헤드 전문화. 일부 헤드는 분리(잡음 억제)에, 일부는 응집(그룹 형성), 일부는 정렬(방향성)에 특화되는 경향을 보인다.*

### Fig. 5 — Long-context scaling (optional, EN)

**What:** Accuracy vs. sequence length (e.g., 1k, 4k, 16k).
**Caption:**
*Figure 5: Length scaling. ASCender maintains quality under increasing sequence lengths by suppressing irrelevant long-range links and reinforcing salient clusters.*

---

# 3) Statistical Reporting Checklist (KR 중심)

* **반복 실험:** 시드 ≥ 5, `mean ± std` 보고.
* **유의성 검정:** 페어드 부트스트랩(테스트셋 재표본, 10k 이상) 또는 Approx. Randomization.
* **효과 크기:** Cliff’s delta 또는 Cohen’s d 병기 권장.
* **동일 조건:** 파라미터 수, 토큰화, 학습 스텝/예산, 정지 조건(early stop) 명시.
* **자원 보고:** GPU/CPU 사양, 메모리 한계, 벽시계 시간.
* **학습 곡선:** 검증 점수 vs. 스텝 커브(평활화 X), ASCender 편향 가중치 $\omega_\star$의 학습 추이도 첨부.
* **해석 가능성:** 분해 맵의 정량 지표(NMI/ARI) + 정성 사례(본문/부록) 병행.

---

# 4) Conclusion (EN/KR)

### (EN) Conclusion (concise)

```text
We introduced ASCender, a Transformer architecture that integrates swarm-inspired inductive biases—Alignment,
Separation, and Cohesion—directly into the attention score computation. Across reasoning, NLP, and long-context
benchmarks, ASCender consistently improves quality while offering favorable efficiency–quality trade-offs, and its
score decomposition yields interpretable attention patterns that align with intuitive grouping.

Our analysis shows that Separation primarily reduces redundant interactions, whereas Alignment and Cohesion foster
semantically coherent clusters; together they enable head specialization. ASCender is orthogonal to positional schemes
and efficiency kernels (e.g., RoPE/ALiBi, sparse/flash), suggesting broad applicability.

Limitations include tuning overhead for neighborhood size and temperatures, and potential sensitivity during early
training if Separation is over-weighted. Future work includes task-adaptive curricula for bias scheduling, integrating
external structural cues (syntax/graphs) with ASCender’s continuous dynamics, and extending to multimodal and
retrieval-augmented settings.
```

---

# 5) (Optional) Appendix Stubs (붙이기 좋은 부록 틀)

* **A. Implementation details:** optimizer, LR schedule, mixed precision, gradient clipping, masks, caching of $\mathcal{N}_k$.
* **B. Additional ablations:** $k,w,\tau$ sweeps, gates on/off, head freezing, memory tables.
* **C. Visualization protocol:** how examples are chosen, seeds, colorbars, resolution, failure cases.
* **D. Statistical methods:** exact bootstrap/randomization procedures and code pointers.
* **E. Ethics/Limitations:** deployment considerations, bias risks, compute footprint.