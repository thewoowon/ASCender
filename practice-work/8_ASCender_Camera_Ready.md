좋습니다. 바로 쓰실 수 있게 **카메라 레디 체크리스트**를 **영문/국문**으로 깔끔하게 정리했습니다. (필요 시 템플릿용 `.tex` 스니펫도 추가해드릴게요.)

# Camera-Ready Checklist (EN)

## 1) Formatting & Compliance

* [ ] Use official template (class, margins, fonts); no page/figure limit violations.
* [ ] All fonts embedded; no Type-3 fonts; PDF/A not required unless venue says so.
* [ ] Line numbers removed (if required); final page numbers as per template.
* [ ] Title/capitalization consistent; abstract within word limit.
* [ ] References style matches venue; all DOIs/years present; no broken citations (“?,” “NaN”).
* [ ] Cross-refs compile: figures/tables/sections/equations/hyperlinks.

## 2) Authorship & Disclosures

* [ ] Final author names, affiliations, ordering confirmed.
* [ ] Corresponding author marked; contact email(s) valid.
* [ ] ORCID IDs (if requested) included.
* [ ] Acknowledgments restored (remove anonymization); funding/grants listed.
* [ ] Competing interests/conflicts disclosed as per venue form.

## 3) Ethics & Compliance

* [ ] Human/animal subjects statements included; IRB/IACUC approval IDs if applicable.
* [ ] Consent details for data collection/sharing (if any).
* [ ] Sensitive data handling, de-identification, and access policy described.
* [ ] Dataset licenses stated; redistribution rights verified; third-party figures/quotes permissions obtained.
* [ ] Broader Impacts / Ethics statement present if required.

## 4) Reproducibility (Artifacts & Docs)

* [ ] Code release plan: repo URL, commit hash, tag, license (e.g., MIT/Apache-2.0).
* [ ] Exact configs (YAML), seeds, hyperparams, schedules; training/inference scripts.
* [ ] Environment files (requirements/conda), CUDA/cuDNN versions; optional Dockerfile.
* [ ] Model checkpoints + checksum; weight license clarified (can differ from code).
* [ ] README with step-by-step reproduction; wall-clock & memory notes; expected metrics ± std.
* [ ] Artifact metadata: dataset versions, preprocessing, tokenizer, prompts/templates.

## 5) Results Integrity

* [ ] Means ± std over ≥5 seeds; significance tests (paired bootstrap or AR).
* [ ] Same training budget across baselines; token counts/steps reported.
* [ ] Hyperparameter search space and selection criteria stated.
* [ ] Any manual selection of examples clearly labeled as illustrative.

## 6) Figures, Tables, Equations

* [ ] Vector graphics (PDF/SVG) where possible; raster ≥ 300 dpi.
* [ ] Color-blind-safe palettes; not color-only encoding; legible fonts ≥ 8 pt.
* [ ] Axes units, legends, error bars; table captions self-contained.
* [ ] Equation numbering consistent; symbols/notation glossary (if needed).

## 7) Accessibility & Readability

* [ ] Alt-text or descriptive captions for key figures (if venue allows).
* [ ] Avoid tiny heatmap text; adequate contrast.
* [ ] Consistent terminology (Alignment/Separation/Cohesion names, abbreviations).

## 8) Venue Forms & Metadata

* [ ] Camera-ready form submitted (copyright/licensing).
* [ ] PDF metadata (Title, Authors, Keywords) set.
* [ ] Artifact/Code/Dataset “badges” or checklists completed (NeurIPS/ICLR style).
* [ ] CO2/compute reporting included if requested (GPU type, hours, energy estimate).

## 9) Packaging & Submission

* [ ] Single PDF compiles cleanly from source; no external network calls in build.
* [ ] Supplementary material (appendix, extra figs, videos) within size/type limits.
* [ ] Names/links stable (short, permanent); QR codes optional but not required.

## 10) ASCender-Specific Items

* [ ] Release **score decomposition** tools (base vs. align/sep/coh) and example notebooks.
* [ ] Provide **bias weights logs** ($\omega_\star$) and **neighborhood stats** (top-k, windows).
* [ ] Include **ablation configs**: A-only, S-only, C-only, A+S, A+C, S+C.
* [ ] Document **bias schedule** (when Separation turns on), temperatures, gating.
* [ ] Flash/sparse compatibility notes (pre-softmax addend integration).
* [ ] Failure cases: over-clustering/under-attention examples with analysis.

---

# 카메라 레디 체크리스트 (KR)

## 1) 포맷 & 규정

* [ ] 공식 템플릿/클래스 사용, 여백/폰트/분량 준수.
* [ ] 폰트 임베딩 완료, Type-3 폰트 없음.
* [ ] 라인 번호 제거(요구 시), 최종 페이지 번호 표기 준수.
* [ ] 제목/대소문자/초록 분량 확인.
* [ ] 참고문헌 형식 일치, DOI/연도 누락 없음, 깨진 인용 없음.
* [ ] 그림/표/식/절 하이퍼링크 정상.

## 2) 저자 정보 & 공개

* [ ] 저자명/소속/순서 확정.
* [ ] 교신저자 지정, 이메일 유효.
* [ ] ORCID(요청 시) 기입.
* [ ] 익명성 해제 후 사사(Ack) 복구, 연구비/과제번호 기재.
* [ ] 이해상충/공익성 고지 제출.

## 3) 윤리 & 준법

* [ ] IRB/IACUC 등 승인/면제 명시, 사람/동물 대상 연구 서술.
* [ ] 데이터 수집·공유 동의 및 비식별화/접근정책 기술.
* [ ] 데이터셋 라이선스·재배포 권리 확인, 3자 자료 사용 허가 확보.
* [ ] 윤리/사회적 영향(요구 시) 섹션 포함.

## 4) 재현성(산출물)

* [ ] 코드 공개: 저장소 URL/커밋 해시/태그/라이선스 명시.
* [ ] 설정파일(YAML), 시드, 하이퍼파라미터, 스케줄, 학습/추론 스크립트.
* [ ] 환경 파일(요구사항/conda), CUDA/cuDNN 버전, 선택적으로 Dockerfile.
* [ ] 체크포인트와 해시, 가중치 라이선스 명확화.
* [ ] README: 재현 절차, 시간/메모리 가이드, 기대 지표(±표준편차).
* [ ] 데이터셋 버전/전처리/토크나이저/프롬프트 기록.

## 5) 결과 신뢰성

* [ ] ≥5회 반복 평균±표준편차 및 유의성 검정 보고.
* [ ] 베이스라인과 동일한 학습 예산(토큰/스텝) 보장.
* [ ] 탐색 범위와 선택 기준 공개.
* [ ] 사례 선택(수기 큐레이션)은 예시임을 명시.

## 6) 그림/표/수식

* [ ] 벡터 그래픽 우선(PDF/SVG), 비트맵은 300 dpi 이상.
* [ ] 색맹 안전 팔레트, 색상만으로 정보 인코딩 금지.
* [ ] 축 단위/범례/오차막대 명확, 표 캡션 자가설명적.
* [ ] 수식 번호/기호/표기 일관, 필요 시 기호표 제공.

## 7) 접근성 & 가독성

* [ ] 주요 그림 대체텍스트 또는 설명적 캡션(가능한 경우).
* [ ] 글자 크기/대비 확보, 과도한 미시 텍스트 지양.
* [ ] 용어 일관(Alignment/Separation/Cohesion 약어 포함).

## 8) 제출 양식 & 메타데이터

* [ ] 카메라 레디 저작권/라이선스 양식 제출.
* [ ] PDF 메타데이터(제목/저자/키워드) 설정.
* [ ] 코드/데이터/아티팩트 배지·체크리스트 완료.
* [ ] 연산/탄소 보고(요청 시): GPU, 시간, 에너지 추정.

## 9) 패키징 & 업로드

* [ ] 소스에서 무오류 컴파일, 외부 네트워크 의존 없음.
* [ ] 부록/부가자료(그림/영상) 용량·형식 제한 준수.
* [ ] 링크는 안정적(퍼머링크/커밋 해시), QR은 선택.

## 10) ASCender 전용 점검

* [ ] **점수 분해 도구**(base vs. align/sep/coh)와 예제 노트북 공개.
* [ ] **바이어스 가중치 로그**($\omega_\star$)와 **이웃 통계**(top-k, 창) 제공.
* [ ] **어블레이션 설정**(A만/S만/C만, 조합) 포함.
* [ ] **바이어스 스케줄**(분리 활성화 시점), 온도, 게이팅 상세 기술.
* [ ] Flash/희소 결합 가이드(softmax 전 가산 방식) 명시.
* [ ] 실패 사례(과응집/저주의)와 원인 분석 포함.

---

원하시면 위 항목에 맞춘 **LaTeX 스니펫**(Ack/Disclosure/Ethics/Artifacts/CO2 템플릿)과 **메타데이터 세팅 스크립트**(e.g., `pdfx`, `hyperref` 옵션)도 만들어 드릴게요.
