좋습니다. 바로 \*\*Experiments 섹션(설계 + 프로토콜)과 PyTorch 레퍼런스 구현(간단판)\*\*을 영어/한국어로 드릴게요. 이후에 원하시면 결과 표 템플릿/그림 가이드까지 붙여서 완성해 드리겠습니다.

# Experiments

## (EN) Research Questions

* **RQ1 (Accuracy):** Do ASCender biases improve task performance over strong Transformer baselines?
* **RQ2 (Efficiency):** Do biases reduce irrelevant interactions and wall-clock/FLOPs under equal quality?
* **RQ3 (Interpretability):** Do decomposed score maps $(\text{base}, \text{align}, \text{sep}, \text{coh})$ align with human-intuitive grouping?
* **RQ4 (Compatibility):** Are gains orthogonal to positional schemes (ALiBi/RoPE) and efficiency kernels (sparse/flash/linear)?

## (KR) 연구 질문

* **RQ1 (정확도):** ASCender 편향이 강력한 기본 Transformer 대비 성능을 향상시키는가?
* **RQ2 (효율성):** 동일 품질 기준에서 불필요 상호작용·FLOPs·시간을 줄이는가?
* **RQ3 (해석 가능성):** 점수 분해맵이 사람이 직관하는 군집·관계와 부합하는가?
* **RQ4 (호환성):** 위치 인코딩·효율 커널과 **직교적**으로 결합 가능한가?

---

## (EN) Benchmarks & Metrics

* **Reasoning/Math:** GSM8K, MATH (Acc / EM; CoT-free primary, CoT as appendix).
* **NLP Understanding:** GLUE, SuperGLUE (task metrics as standard).
* **Long-context:** Long Range Arena / LongBench / SCROLLS (task-specific; plus latency & memory).
* **Vision/Text-grouping (optional):** Object-centric grouping or segmentation-affinity proxies (NMI/ARI vs. pseudo labels).
* **Efficiency:** Attention FLOPs, peak memory, tokens/sec; report with/without FlashAttention or sparse windows.

## (KR) 벤치마크·지표

* **추론/수학:** GSM8K, MATH (정확도/EM; 기본은 CoT 미사용, 부록에 CoT).
* **언어 이해:** GLUE, SuperGLUE(표준 지표).
* **장문:** LRA / LongBench / SCROLLS(과제별 지표 + 지연/메모리).
* **그룹핑(선택):** 객체/슬롯 기반 군집의사라벨과 NMI/ARI.
* **효율:** Attention FLOPs, 피크 메모리, 처리속도(tokens/sec), Flash/희소 결합 전후.

---

## (EN) Baselines

1. **Vanilla Transformer** (+ RoPE or ALiBi).
2. **Efficiency variants:** Longformer/BigBird (sparse), Performer/Linear (kernelized), FlashAttention (kernel fusion).
3. **Structure priors:** relative distance bias; (optional) syntax/graph-aware bias.

**Our models:**

* **ASCender-Base:** base + A/S/C (ours).
* **Compatibility runs:** ASCender + RoPE/ALiBi; ASCender + sparse/flash.

## (KR) 비교 대상

1. **기본:** 바닐라 Transformer(+ RoPE 또는 ALiBi).
2. **효율:** Longformer/BigBird(희소), Performer/Linear(커널), FlashAttention(커널 융합).
3. **구조 편향:** 상대 거리 바이어스, (선택) 구문/그래프 편향.

**우리 모델:**

* **ASCender-Base:** 기본+정렬/분리/응집.
* **결합 실험:** ASCender + RoPE/ALiBi, ASCender + 희소/Flash.

---

## (EN) Training Setup

* **Model sizes:** Small/Medium (to iterate quickly), then Large for confirmatory.
* **Optimization:** AdamW; lr warm-up; cosine decay.
* **ASCender schedule:** enable **Alignment/Cohesion** from step 0; **Separation** after warm-up (e.g., 10–20% steps).
* **Neighborhood:** semantic top-$k$ within a local window $w$ + few globals; tune $k\in\{8,16,32\}$, $w\in\{64,128,256\}$.
* **Temperatures:** $\tau_{\text{sep}},\tau_{\text{coh}}\in\{0.5,1,2\}$; score temperature $\tau_{\text{score}}\in\{0.7,1.0\}$.
* **Gates/weights:** $\omega_\star$ start small (e.g., 0.1) → learnable; clip norms; row-wise bias normalization.

## (KR) 학습 설정

* **모델 크기:** Small/Medium에서 탐색 후 Large 확인.
* **최적화:** AdamW, 워밍업, 코사인 감쇠.
* **ASCender 스케줄:** 정렬/응집은 초반부터, **분리**는 워밍업 이후 활성화.
* **이웃:** 국소 창 $w$+글로벌 소수, $k\in\{8,16,32\}$, $w\in\{64,128,256\}$.
* **온도:** $\tau_{\text{sep}},\tau_{\text{coh}}\in\{0.5,1,2\}$; $\tau_{\text{score}}\in\{0.7,1.0\}$.
* **게이트/가중치:** $\omega_\star$ 작게 시작→학습; 정규화·클리핑 적용.

---

## (EN) Ablations & Analyses

* Remove each bias (A/S/C) and all combinations.
* Vary $k, w, \tau$, and whether neighborhoods reuse base top-$k$.
* Head specialization: fix one bias/head vs. all heads shared.
* **Interpretability:** visualize $S^{\text{base}}$ and each $\tilde\beta$; measure cluster quality (NMI/ARI) vs. labels or pseudo-labels.
* **Significance:** 10× runs, report mean±std; paired bootstrap and approximate randomization tests.

## (KR) 어블레이션·분석

* 편향 항 제거 실험(단일·조합), $k/w/\tau$ 변화, top-$k$ 재사용 여부.
* 헤드 특화 vs. 공유.
* **해석:** $S^{\text{base}}$ 및 $\tilde\beta$ 시각화, 군집 지표(NMI/ARI).
* **통계:** 10회 반복, 평균±표준편차, 부트스트랩/랜덤화 검정.

---

## (EN) Reproducibility

* Seeds $\{1,2,3,5,7\}$, deterministic ops where feasible.
* Report hardware, peak memory, tokens/sec.
* Release config files and scripts; log bias weights $\omega_\star$ over time.

## (KR) 재현성

* 시드 다중, 결정적 연산 범위 내 사용.
* 하드웨어/메모리/속도 보고.
* 설정·스크립트 공개, $\omega_\star$ 학습 로그 공개.

---

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

---

원하시면 다음 턴에서 **Results 표 템플릿(EN/KR)**, **그림 가이드(주의 맵 분해, 군집 지표)**, 그리고 **논문 결론(Conclusion) 초안**까지 한 번에 정리해 드릴게요.
