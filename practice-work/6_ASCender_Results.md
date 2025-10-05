# Results — templates, figures, and conclusion (EN/KR)

아래는 **결과 표 템플릿**, **그림(시각화) 가이드**, 그리고 **결론(Conclusion) 초안**입니다. 수치가 확정되면 그대로 채워 넣기만 하면 되도록 구성했어요.

---

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

### (KR) 효율/지연/메모리

```text
표 2: 효율 지표(동일 품질 목표선에서 측정).
---------------------------------------------------------------------------
모델                 | Attention FLOPs | Peak Memory (GB) | Tokens/sec | Latency (ms)
---------------------|-----------------|------------------|------------|-------------
Transformer (base)   |      XXXX       |       X.XX       |    XXXX    |    XXX
ASCender (ours)      |      XXXX       |       X.XX       |    XXXX    |    XXX
ASC + Sparse/Flash   |      XXXX       |       X.XX       |    XXXX    |    XXX
---------------------------------------------------------------------------
주: 동일 파라미터 수/배치/시퀀스 길이/정확도 범위 내 맞춘 공정 비교.
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

### (KR) 해석 가능성/군집 지표

```text
표 4: 주의 맵 분해 기반 군집 품질 (의사 라벨 또는 보조 태스크 라벨 대비).
---------------------------------------------------------------------------
모델                 | NMI ↑ | ARI ↑ | Calinski-Harabasz ↑ | Silhouette ↑
---------------------|-------|-------|---------------------|--------------
Transformer (base)   |  X.XX | X.XX  |        XXXX         |     X.XX
ASCender (ours)      |  X.XX | X.XX  |        XXXX         |     X.XX
ASC (ablations)      |  X.XX | X.XX  |        XXXX         |     X.XX
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

### (KR) 결론 (간결판)

```text
본 연구는 Boids의 정렬·분리·응집 원리를 Attention 점수에 직접 주입하는 ASCender를 제안하였다.
추론·언어·장문 벤치마크에서 ASCender는 품질을 향상시키면서 효율–품질 균형을 개선하며, 점수 분해 맵은
사람이 이해 가능한 군집 구조를 드러낸다. 분석 결과, 분리는 중복 상호작용을 억제해 효율에 기여하고,
정렬·응집은 의미적 군집을 형성해 성능을 끌어올리며, 헤드별 역할 분화가 관찰되었다. 또한 RoPE/ALiBi,
희소/Flash 계열과 직교적으로 결합 가능해 적용 범위가 넓다.

한계로는 이웃 크기·온도 등 하이퍼파라미터 튜닝 부담과 초반 학습에서 분리 항 과대 시 불안정성이 있다.
향후 연구로는 태스크 적응형 바이어스 스케줄링, 구문/그래프 등 외부 구조 단서와의 결합, 멀티모달·
검색 결합 환경으로의 확장을 제안한다.
```

---

# 5) (Optional) Appendix Stubs (붙이기 좋은 부록 틀)

* **A. Implementation details:** optimizer, LR schedule, mixed precision, gradient clipping, masks, caching of $\mathcal{N}_k$.
* **B. Additional ablations:** $k,w,\tau$ sweeps, gates on/off, head freezing, memory tables.
* **C. Visualization protocol:** how examples are chosen, seeds, colorbars, resolution, failure cases.
* **D. Statistical methods:** exact bootstrap/randomization procedures and code pointers.
* **E. Ethics/Limitations:** deployment considerations, bias risks, compute footprint.

---

필요하시면 **그림 캡션까지 포함된 LaTeX 표/그림 코드**(ICLR/NeurIPS 템플릿 호환)와, **카메라 레디 체크리스트**(저자구성/라이선스/데이터카드)도 붙여드릴게요.
다음으로는 **Discussion** 또는 **Limitations & Broader Impacts** 섹션을 이어갈까요?
