좋아, “왜 이런 수식과 바이어스들을 쓰는가?”에 대해 **(A) 선행연구 근거**와 **(B) 간결하지만 빡센 원리적 유도** 두 갈래로 정리해 줄게. 아래 표기는 너의 row-vector convention과 위 섹션의 기호에 맞춰 뒀어.

---

# (A) 선행연구로 본 정당성

### 1) “pre-softmax 가감(加減)으로 attention 점수를 형태화한다”는 관행

* **상대/절대/회전 위치 바이어스**는 모두 $S_{ij}$에 **가산 항**을 더하는 방식으로 정착돼 있음.
  대표적으로 **Relative Position Representations**(Shaw et al., 2018), **RoPE**(Su et al., 2021), **ALiBi**(Press et al., 2021)는 $q_i^\top k_j/\sqrt{d_k}$에 거리/위치 의존 항을 **그대로 더**해서 분포를 바꿈. 이는 네가 정의한 $\beta^{(\text{other})}$와 정확히 같은 자리다. ([arXiv][1]) ([arXiv][2]) ([arXiv][3], [Ofir][4], [OpenReview][5])

> ⇒ 따라서 $\beta^{\text{align}},\beta^{\text{sep}},\beta^{\text{coh}}$를 **pre-softmax addend**로 더하는 설계는, 기존 위치 바이어스 계열과 **완전히 동일한 수학적 위치**에 놓인다.

---

### 2) $\beta^{\text{coh}}$: **커널-평균(centroid) 끌림**은 고전적

* $w_{ij}^{\tau}\propto\exp(-\|z_i-z_j\|^2/\tau^2)$ 로 **가중 중심** $c_i$를 만들고 $-\|z_j-c_i\|^2$ 보너스를 주는 건, **Mean-Shift**(커널 밀도 기울기 따라 mode로 수렴)와, **그래프 라플라시안/매니폴드 학습**(로컬 스무딩, $w_{ij}$ 가우시안 가중)에서 쓰는 표준적 유도와 일치한다. ([courses.csail.mit.edu][6], [Semantic Scholar][7]) ([MIT Press Direct][8], [NeurIPS Papers][9]) ([NeurIPS Proceedings][10], [NeurIPS Papers][11])

> ⇒ “로컬 이웃의 커널 가중 평균으로 당긴다”는 응집 항은 **밀도추정·매니폴드 스무딩의 정석적 기법**을 attention 점수에 접목한 것.

---

### 3) $\beta^{\text{sep}}$: **중복 억제/다양성 유도**의 확립된 원리

* 표현 중복을 줄이고 **서로 너무 비슷한 이웃을 밀어내는** 설계는, \*\*DPP(Determinantal Point Process)\*\*의 **반발(repulsion)** 모형, \*\*대조학습(contrastive)\*\*의 음수-끌어내기, **Barlow Twins**의 **redundancy reduction**과 같은 정식화와 철학이 같다. 네 식의 $\phi^{\text{red}}_{ij}=w_{ij}\max(0,a_{ij}-\delta)$는 “가깝고(큰 $w$) 유사한(큰 $a$)” 쌍을 **패널티**로 본다는 점에서 이 계열과 호응한다. ([arXiv][12], [alexkulesza.com][13]) ([NeurIPS Proceedings][14]) ([arXiv][15])

> ⇒ **과밀/중복 억제**는 표현학습 문헌에서 성능·안정성을 위해 널리 쓰는 **정당한 귀납 편향**.

---

### 4) $\beta^{\text{align}}$: **방향 통계(direction statistics)** 기반의 게이팅

* 이웃의 단위벡터 $\{\tilde{k}_l\}$ 평균의 **결과벡터 길이**(mean resultant length)는 \*\*농도(concentration)\*\*의 통계량이고, **von Mises/VMF** 분포의 **$\kappa$**(집중도)와 단조 관계를 가진다. 너의 $\gamma_{\text{align}}(i)=\sigma(\alpha\,\mathrm{Var}[\tilde{k}_l])$는 사실상 “결과벡터 길이가 클수록(=분산 작을수록) 정렬을 강하게”라는 **표준적 게이팅**의 부호만 바꾼 형태다. ([위키백과][16], [구글 도서][17])

> ⇒ **이웃 방향의 일관성**이 높을수록 정렬 항을 키우는 설계는 **방향 통계의 교과서적 근거**를 갖는다.

---

### 5) 구현 상의 호환성

* 네가 적은 “**pre-softmax addend**로 융합”은 **FlashAttention**(정확주의 타일링 커널), **Longformer/BigBird**(윈도+소수 글로벌)와 **완벽 호환**된다. 바이어스는 점수 $S$에 더하는 항이라 커널 계산과 **직교적**으로 결합된다. ([arXiv][18]) ([arXiv][19]) ([NeurIPS Proceedings][20], [arXiv][21])

---

# (B) 원리적 유도(sketch)

**목표:** 쿼리 $i$가 키 $j$를 선택하는 분포 $A_{i:}$를 “유용도”에 맞춰 갖게 하되, 엔트로피로 과신을 제어.
다음 **맥스엔트(max-entropy) 원리** 기반의 에너지 최소화를 보자.

* **에너지/효용 정의**

  $$
  U_{ij}
  \;=\;
  \underbrace{\frac{q_i^\top k_j}{\sqrt{d_k}}}_{\text{기본 유사도}}
  \;+\;
  \underbrace{\omega_{\text{align}}\,r^{\text{align}}_{ij}}_{\text{이웃 평균방향 정렬}}
  \;+\;
  \underbrace{\omega_{\text{coh}}\frac{r^{\text{coh}}_{ij}}{\tau_{\text{coh}}}}_{\text{커널 중심 응집}}
  \;-\;
  \underbrace{\omega_{\text{sep}}\,\eta_i\,\phi^{\text{red}}_{ij}}_{\text{과밀·중복 억제}}
  $$

* **라그랑주안:**
  $\displaystyle \min_{A_{i:}\in\Delta}\; \mathcal{E}_i(A)= -\sum_j A_{ij}U_{ij} \;+\; \tau_{\text{score}}\sum_j A_{ij}\log A_{ij}$

* **정리:** 정규화 제약 $\sum_j A_{ij}=1$ 하에서 1차 조건을 풀면

  $$
  A_{ij}
  \;=\;
  \frac{\exp\!\big(U_{ij}/\tau_{\text{score}}\big)}{\sum_{j'} \exp\!\big(U_{ij'}/\tau_{\text{score}}\big)}
  \;=\;
  \text{softmax}_j\!\Big(\tfrac{q_i^\top k_j}{\sqrt{d_k}} + \sum_\star \omega_\star \beta^\star_{ij}\Big)
  $$

  즉, **pre-softmax 가산 항** $\beta^\star_{ij}$는 **Gibbs 분포의 로그-퍼텐셜**로 해석된다(확률 모형 관점).
  이때 $\tau_{\text{score}}$는 **온도/엔트로피 조절**이고, 행별 표준화 $\tilde\beta$는 **드리프트 제거**(수렴성 향상)에 해당.

* **미분 직관:**
  $\nabla_{k_j} \mathbb{E}_{j\sim A_i}[U_{ij}]$를 보면
  정렬은 $u_i$ 방향으로 $k_j$를 돌리고, 응집은 $z_j$를 $c_i$로 당기며, 분리는 $a_{ij}$가 큰 근접 이웃을 밀어낸다(온도·게이트로 조절).

> 요컨대, $\beta$ 들은 **에너지 기반 attention**의 **합리적 로그-퍼텐셜**로서, 각각 **정렬-응집-분리**라는 해석 가능하고 검증된 귀납 편향을 부여한다.

---

## 설계 선택에 대한 추가 코멘트(세부 타당화)

* **정규화:** 행별 $z$-정규화는 바이어스 스케일의 **드리프트/폭주 방지**에 효과적(학습 안정화). (일반적 전처리 원리)
* **커널 폭 $\tau$:** $\tau^2$ 분모는 RBF/Laplacian 문헌과 일관. 그래프 학습·NL-means에서도 동일한 가우시안 커널을 쓴다. ([NeurIPS Proceedings][10], [NeurIPS Papers][11]) ([iro.umontreal.ca][22])
* **윈도우+글로벌:** Longformer/BigBird가 검증했듯, 로컬 윈도우에 **소수 글로벌 토큰**을 섞는 설계는 계산·성능 균형이 좋고, 네 $\mathcal{N}_k(i)$ 제한과 궁합이 맞다. ([arXiv][19]) ([NeurIPS Proceedings][20], [arXiv][21])
* **커널-중심 vs. 라플라시안:** $\beta^{\text{coh}}$의 $r^{\text{coh}}_{ij}\!=\!-\|z_j-c_i\|^2$는 mean-shift류의 **모드 수렴** 직관과, 그래프 정규화의 **로컬 매끄러움**(인접하면 비슷해야 함) 원리를 동시에 취한다. ([courses.csail.mit.edu][6], [Semantic Scholar][7]) ([MIT Press Direct][8])
* **중복 억제 임계 $\delta$:** DPP/대조학습/Barlow Twins가 모두 “**너무 비슷한 것끼리는 떨어뜨려라**”는 동일 철학을 취하므로, $a_{ij}>\delta$에서만 페널티를 주는 건 자연스럽다. ([arXiv][12]) ([NeurIPS Proceedings][14]) ([arXiv][15])
* **정렬 게이트:** 이웃 단위벡터의 **결과벡터 길이 $R$** 는 분산과 역상관, VMF의 **$\kappa$** 와 단조. $\mathrm{Var}$ 기반 $\gamma_{\text{align}}$은 **$R$ 기반 게이팅**과 동일한 정보를 담는다. ([위키백과][16], [구글 도서][17])

---

## 요약(한 줄씩)

* **자리에 대한 정당성:** $\beta$ 들은 위치 바이어스와 같은 **pre-softmax addend**로, Transformer 문헌의 표준 자리. ([arXiv][1])
* **응집:** 커널-평균 끌림 = mean-shift/라플라시안 스무딩의 정석. ([courses.csail.mit.edu][6], [MIT Press Direct][8])
* **분리:** 중복 억제 = DPP/대조/Barlow Twins의 반발·다양성 원리. ([arXiv][12], [NeurIPS Proceedings][14])
* **정렬:** 이웃 방향의 일관성으로 게이트 = 방향통계의 농도(κ)와 직결. ([위키백과][16], [구글 도서][17])
* **구현:** FlashAttention·윈도우 계열과 **직교적** 융합 가능. ([arXiv][18], [NeurIPS Proceedings][20])

---

[1]: https://arxiv.org/abs/1803.02155?utm_source=chatgpt.com "Self-Attention with Relative Position Representations"
[2]: https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com "RoFormer: Enhanced Transformer with Rotary Position Embedding"
[3]: https://arxiv.org/abs/2108.12409?utm_source=chatgpt.com "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
[4]: https://ofir.io/train_short_test_long.pdf?utm_source=chatgpt.com "TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR ..."
[5]: https://openreview.net/forum?id=R8sQPpGCv0&utm_source=chatgpt.com "Train Short, Test Long: Attention with Linear Biases ..."
[6]: https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf?utm_source=chatgpt.com "Mean shift: a robust approach toward feature space analysis"
[7]: https://www.semanticscholar.org/paper/Mean-Shift%3A-A-Robust-Approach-Toward-Feature-Space-Comaniciu-Meer/74f4ecc3e4e5b91fbb54330b285ed5214afe2001?utm_source=chatgpt.com "Mean Shift: A Robust Approach Toward Feature Space ..."
[8]: https://direct.mit.edu/neco/article/15/6/1373/6730/Laplacian-Eigenmaps-for-Dimensionality-Reduction?utm_source=chatgpt.com "Laplacian Eigenmaps for Dimensionality Reduction and ..."
[9]: https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering?utm_source=chatgpt.com "Laplacian Eigenmaps and Spectral Techniques for ..."
[10]: https://proceedings.neurips.cc/paper/2506-learning-with-local-and-global-consistency.pdf?utm_source=chatgpt.com "Learning with Local and Global Consistency"
[11]: https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency?utm_source=chatgpt.com "Learning with Local and Global Consistency"
[12]: https://arxiv.org/abs/1207.6083?utm_source=chatgpt.com "Determinantal point processes for machine learning"
[13]: https://www.alexkulesza.com/pubs/dpps_fnt12.pdf?utm_source=chatgpt.com "Determinantal Point Processes for Machine Learning Contents"
[14]: https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf?utm_source=chatgpt.com "Supervised Contrastive Learning"
[15]: https://arxiv.org/abs/2103.03230?utm_source=chatgpt.com "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
[16]: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution?utm_source=chatgpt.com "Von Mises–Fisher distribution"
[17]: https://books.google.es/books?id=PTNiCm4Q-M0C&utm_source=chatgpt.com "Directional Statistics - Kanti V. Mardia, Peter E. Jupp"
[18]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[19]: https://arxiv.org/abs/2004.05150?utm_source=chatgpt.com "Longformer: The Long-Document Transformer"
[20]: https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html?utm_source=chatgpt.com "Big Bird: Transformers for Longer Sequences"
[21]: https://arxiv.org/pdf/2007.14062?utm_source=chatgpt.com "Big Bird: Transformers for Longer Sequences"
[22]: https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf?utm_source=chatgpt.com "A non-local algorithm for image denoising"
