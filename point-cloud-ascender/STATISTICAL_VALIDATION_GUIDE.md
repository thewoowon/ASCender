# 📊 Statistical Validation 실행 가이드

졸업 논문 제출을 위한 통계적 검증 실험 실행 방법

---

## 🚀 빠른 실행 (추천)

### AWS 병렬 실행 (2-3시간, ~$19)

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender
bash RUN_EXPERIMENTS_AWS.sh
```

**이 스크립트는 자동으로:**
- ✅ S3 버킷 생성 및 데이터 업로드
- ✅ 12개 GPU 인스턴스에서 병렬 학습
- ✅ 결과 자동 수집 및 통계 분석
- ✅ 완료 후 인스턴스 자동 종료

**예상 소요 시간:**
- 설정: ~5분
- 학습: ~2-3시간 (병렬)
- 총 시간: **~3시간**

**예상 비용:** ~$19

---

## 💻 로컬 실행 (순차, 24-36시간)

GPU가 있는 로컬 머신에서 실행:

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# 전체 실험 (12개, 24-36시간)
python experiments/run_statistical_validation.py --all

# h32만 실행 (6개, 12-18시간)
python experiments/run_statistical_validation.py --model-size h32

# h80만 실행 (6개, 12-18시간)
python experiments/run_statistical_validation.py --model-size h80
```

**완료 후 통계 계산:**
```bash
python experiments/compute_statistics.py
```

---

## 📊 결과 확인

### 통계 요약 보기

```bash
# 테이블 형식
cat results/statistical_validation/summary_table.txt

# JSON 형식
cat results/statistical_validation/summary.json | python -m json.tool
```

### 예상 출력

```
================================================================================
STATISTICAL VALIDATION RESULTS
================================================================================

h32 Results:
--------------------------------------------------------------------------------
  Baseline:  74.84% ± 1.23%
  ASCender:  77.05% ± 0.98%
  Δ Acc:     +2.21%
  p-value:   0.0123 *
  Result:    ASCender wins (statistically significant)

h80 Results:
--------------------------------------------------------------------------------
  Baseline:  83.02% ± 0.87%
  ASCender:  83.71% ± 1.05%
  Δ Acc:     +0.69%
  p-value:   0.1234 (n.s.)
  Result:    No significant difference

================================================================================
Significance levels: *** p<0.001, ** p<0.01, * p<0.05, (n.s.) not significant
================================================================================
```

---

## 🔍 실험 설계

### 실험 목록 (총 12개)

#### h32 (7.1K parameters)
1. h32_baseline_seed42
2. h32_baseline_seed123
3. h32_baseline_seed456
4. h32_ascender_seed42
5. h32_ascender_seed123
6. h32_ascender_seed456

#### h80 (37.6K parameters)
7. h80_baseline_seed42
8. h80_baseline_seed123
9. h80_baseline_seed456
10. h80_ascender_seed42
11. h80_ascender_seed123
12. h80_ascender_seed456

### 통계적 검증 방법

- **반복 실험:** 각 설정당 3회 (seeds: 42, 123, 456)
- **평가 지표:** Mean ± Standard Deviation
- **유의성 검정:** Paired t-test (양측)
- **신뢰 구간:** 95% CI
- **효과 크기:** Cohen's d

---

## 📝 논문 업데이트

통계 결과가 나오면:

### 1. PAPER_DRAFT.md 업데이트

```markdown
## Results

### ModelNet40 Validation

| Model | Parameters | Baseline | ASCender | Δ Acc | p-value |
|-------|-----------|----------|----------|-------|---------|
| h32   | 7.1K      | 74.84±1.23% | 77.05±0.98% | **+2.21%** | 0.012* |
| h48   | 14.9K     | 79.46% | 77.47% | -1.99% | - |
| h64   | 25.0K     | 82.37% | 81.52% | -0.85% | - |
| h80   | 37.6K     | 83.02±0.87% | 83.71±1.05% | +0.69% | 0.123 |

*Statistical validation performed with 3 random seeds for h32 and h80.*
*p < 0.05 indicates statistically significant difference.*
```

### 2. Methods 섹션 추가

```markdown
## Experimental Protocol

All experiments were conducted using the following protocol:
- **Random seeds:** 42, 123, 456 (3 runs per configuration)
- **Training:** 50 epochs with Adam optimizer (lr=1e-3)
- **Hardware:** NVIDIA T4 GPU, batch size 32
- **Statistical test:** Paired t-test with α=0.05
- **Results:** Mean ± standard deviation across 3 runs
```

---

## ⏱️ 시간 계획

### 내일 제출 스케줄

**총 소요 시간: ~4-5시간**

| 시간 | 작업 | 소요 시간 |
|------|------|----------|
| 0:00 | AWS 실험 시작 | 10분 |
| 0:10 | 대기 (학습 진행) | 3시간 |
| 3:10 | 결과 다운로드 | 5분 |
| 3:15 | 통계 분석 | 2분 |
| 3:17 | 논문 업데이트 (영문) | 30분 |
| 3:47 | 논문 업데이트 (한글) | 30분 |
| 4:17 | 최종 검토 | 30분 |
| 4:47 | **제출 준비 완료** | - |

### 추천 일정

**오늘 저녁:**
- 19:00: AWS 실험 시작
- 22:00: 결과 확인 및 통계 분석
- 23:00: 논문 업데이트
- 24:00: 최종 검토

**내일 아침:**
- 최종 확인 후 제출

---

## 🛠️ 문제 해결

### AWS 실행 중 에러

```bash
# 로그 확인
aws s3 ls s3://ascender-experiments-*/

# 인스턴스 상태 확인
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=ASCender" \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name]' \
    --output table
```

### 비용 초과 방지

```bash
# 실행 중인 인스턴스 확인
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text

# 모든 인스턴스 종료
aws ec2 terminate-instances --instance-ids [instance-ids]
```

### 로컬 실행으로 전환

시간이 부족하면 h32만 실행:

```bash
# h32만 3회 반복 (12시간)
python experiments/run_statistical_validation.py --model-size h32

# h80은 기존 단일 실행 결과 사용
# 논문에 명시: "h80은 계산 제약으로 단일 실행"
```

---

## ✅ 최종 체크리스트

실행 전:
- [ ] 전처리된 데이터 확인 (`data/ModelNet40/preprocessed/`)
- [ ] AWS CLI 설정 확인 (`aws configure list`)
- [ ] 예산 확인 (~$19)
- [ ] 시간 확보 (3시간 대기)

실행 중:
- [ ] 12개 인스턴스 실행 확인
- [ ] S3에 결과 업로드 확인
- [ ] 진행 상황 주기적 체크

실행 후:
- [ ] 12개 결과 파일 확인
- [ ] 통계 분석 완료
- [ ] 논문 업데이트 (영문/한글)
- [ ] AWS 리소스 정리

---

## 📧 긴급 연락

문제 발생 시:
1. 에러 메시지 확인
2. AWS Console 로그 확인
3. 로컬 실행으로 전환 고려

**중요:** 내일 제출이므로 오늘 중 반드시 실험 시작!
