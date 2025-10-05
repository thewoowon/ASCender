# Method — ASCender (English/Korean)

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

원하면 위 수식을 그대로 반영한 **PyTorch(또는 JAX) 레퍼런스 구현**과,
\*\*실험 섹션(benchmarks, metrics, 시각화 프로토콜)\*\*을 바로 이어서 작성해 드릴게요.
