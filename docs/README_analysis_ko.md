# 평가지표 / LOVO / 부트스트랩 스크립트 사용법

기존 파이프라인(`config.py`, `run_experiment.py` 등)이 만든 `results/*.jsonl`을
그대로 입력으로 받습니다. 아래 4개 파일을 파이프라인 폴더에 함께 두세요.

- `data_utils.py` — 공통 로더/필터 (다른 세 스크립트가 여기 의존)
- `metrics.py` — ACC / TBR_C / FBR_C / Δ_AA 계산 + 요약표 + 대조 분석
- `leave_one_verb_out.py` — 소표본 동사 유형 강건성 검증
- `bootstrap_ci.py` — 동사 단위 클러스터 부트스트랩 신뢰구간

---

## 이중 라벨링 체계 (Dual Labeling System)

한국어와 영어의 언어적 차이로 인해 동사 분류 체계가 달라서 **이중 라벨링 체계**를 도입했습니다. 완결동사(그룹 A/C)의 각 동사에 대해 두 층위의 라벨을 부여합니다:

1. **1차 라벨 (`verb_class`)**: 원 논문(Ma & Miyao, 2026)과의 비교를 위해 유지하는 4분류 (`Creation`, `Consumption`, `Change_of_State`, `Motion_to_Goal`).
2. **2차 라벨 (`secondary_class`)**: 한국어 문법 진단 테스트(예: "거의 ~할 뻔했다" 자연스러움 여부, "~하는 데 시간이 걸리다" 허용 여부 등)를 거쳐 부여한 실제 한국어 어휘상 범주 (`완성동사`, `정도달성동사`, `전형적 달성동사`).

두 라벨이 다르게 묶이는 지점(예: Creation = Consumption = 완성동사)은 **"문법 범주는 같지만 의미가 다른 동사 간 비교"**라는 별도의 분석 축으로 활용됩니다.

### 이중 라벨링 관련 모듈별 주요 기능 및 변경사항

#### 1. `data_utils.py`
- `SECONDARY_VERB_CLASSES = ("완성동사", "정도달성동사", "전형적 달성동사")` 상수 정의
- `load_secondary_labels(path)`: 매핑 CSV(`비완결동사_이중라벨링_매핑.csv`)를 읽어 정리 (2차 라벨, 1차 라벨 정규화, 복합표기 충돌 플래그 `primary_class_conflict`, `context_dependent` 문맥의존 플래그 등 추출)
- `attach_secondary_labels(df, secondary_labels)`: `verb` 컬럼 기준으로 조인. 완결동사(A/C)인데 매핑에 없으면 에러를 발생시키며, 1차/2차 라벨이 어긋나면 경고를 출력 (원본 `verb_class` 유지)
- `load_results(..., secondary_labels_path=...)`: `secondary_labels_path` 인자 지정 시 결과 읽기와 2차 라벨 조인까지 한 번에 자동 처리 (미지정 시 `secondary_class`는 NaN으로 남아 기존 로직과 호환)
- `filter_df(...)`: `secondary_class` 필터 추가 (`verb_class`와 독립적으로 AND 결합 가능)

#### 2. `metrics.py`
- `FBR_C` (False Belief Rate): 그룹 C에서 모델이 "거짓"이라고 응답한 비율(미완료 과신 비율) 지표 추가 (`TBR_C`와 대칭)
- `ACC`, `TBR_C`, `FBR_C`, `delta_aa`: `secondary_class` 인자를 지원하도록 확장
- `summarize()`: (model, strategy) 요약표에 `FBR_C` 컬럼 추가
- `summarize_by_secondary_class()`: 2차 라벨(문법범주) 기준 요약표 생성 — "이동 vs 상태변화" 류 비교용
- `semantic_contrast()`: 동일 2차 라벨 안에서 서로 다른 1차 라벨을 비교하는 헬퍼 함수 (예: `semantic_contrast(df, "C", ("Creation", "Consumption"), "완성동사")` 호출 시, 완성동사로 확인된 창조·소비 동사만 추출하여 동일 문법범주 내 순수 의미론적 대조 분석 제공)
- `class_gap()`: `verb_class_gap`을 일반화하여 `class_column='verb_class'` 또는 `'secondary_class'` 기준으로 세부 유형 간 격차 계산 가능 (`verb_class_gap`은 하위호환용 래퍼로 유지)
- CLI 옵션: `--secondary-labels`, `--out-by-secondary-class` 추가

#### 3. `bootstrap_ci.py`
- `bootstrap_metric`, `bootstrap_delta_aa`: `secondary_class` 인자를 받아 2차 라벨 기준 리샘플링 지원
- `run_all()`: `df`에 `secondary_class`가 채워져 있으면(매핑 조인 시) `TBR_C`, `FBR_C`, `Delta_AA`를 2차 라벨 기준 CI로도 자동 계산 (매핑 미결합 시 에러 없이 조용히 1차 라벨만 계산)
- CLI 옵션: `--secondary-labels`, `--secondary-classes` 추가

#### 4. `leave_one_verb_out.py`
- `lovo_for_condition`, `lovo_summary`, `run_lovo`: 2차 라벨(`secondary_class`) 및 `FBR_C` 지표 지원
- CLI 옵션: `--secondary-labels`, `--secondary-classes` 추가

---

## 0. 지표 정의

- `ACC_g` = 그룹 g(A/B/C/D)의 accuracy
- `TBR_C` = 그룹 C(gold=Unknown)에서 모델이 "참"이라고 답한 비율 — 완료 환각률 (Teleological Bias Rate)
- `FBR_C` = 그룹 C(gold=Unknown)에서 모델이 "거짓"이라고 답한 비율 — 미완료 과신률 (False Belief Rate)
- `Δ_AA` = `ACC_D - TBR_C` — 비완결 판단력(그룹 D) 대비 완결 환각 정도의 격차

데이터에서 확인된 그룹별 gold label: A=False, B=True, C=Unknown, D=True.

---

## 1. 평가지표 요약표

```bash
# 기본 1차 라벨 기준 요약
python metrics.py results/main.jsonl \
  --out-summary results/summary.csv \
  --out-by-verb-class results/summary_by_verb_class.csv \
  --out-gap results/verb_class_gap.csv

# 2차 라벨(문법범주) 및 Step 1 대조 분석 포함 요약
python metrics.py results/main.jsonl \
  --secondary-labels 비완결동사_이중라벨링_매핑.csv \
  --out-summary results/summary.csv \
  --out-by-verb-class results/summary_by_verb_class.csv \
  --out-by-secondary-class results/summary_by_secondary_class.csv \
  --out-gap results/verb_class_gap.csv \
  --out-contrast results/contrast_summary.csv
```

- `summary.csv`: (model, strategy)별 ACC_A/B/C/D, TBR_C, FBR_C, Δ_AA
- `summary_by_verb_class.csv`: 그룹 A/C를 완결동사 1차 세부 유형(창조/소비/상태변화/이동)별로 쪼갠 표
- `summary_by_secondary_class.csv`: 그룹 A/C를 한국어 2차 문법범주(완성동사/정도달성동사/전형적 달성동사)별로 쪼갠 표
- `verb_class_gap.csv`: 세부 유형 간 TBR_C 최대-최소 격차 + 사전 기준(15%p) 충족 여부 플래그
- `contrast_summary.csv`: **[Step 1]** 1차/2차 이중 라벨 대조 분석 표 (이동 완성 vs 소비 완성, 이동 달성 vs 소비 완성 등 TBR_C/FBR_C 격차)

기본적으로 `parse_ok=False`인 행은 지표 계산에서 제외됩니다. 포함하려면 `--keep-unparsed`.

---

## 2. Step 2: 사고 과정(Reasoning Log) 정량 카테고리화 및 오류 분석

모델이 출력한 reasoning(생론)을 분석하여 상적 오판(Telic 기준 오적용, 중단 사건 과해석, 경계통과 상 충돌 등)의 원인을 5대 카테고리로 정량 분류합니다.

```bash
# 사고 과정 오류 분류 실행
python analyze_reasoning.py results/main.jsonl \
  --secondary-labels 비완결동사_이중라벨링_매핑.csv \
  --out-summary results/reasoning_summary.csv \
  --out-by-group results/reasoning_by_group.csv \
  --out-classified-jsonl results/reasoning_classified.csv
```

- `reasoning_summary.csv`: (model, strategy)별 오류 유형 카테고리 빈도 및 비율(%)
- `reasoning_by_group.csv`: (model, strategy, group_code)별 오류 분포 표 (특히 Group D 비완결동사 오판 및 Group C FBR_C 과해석 원인 규명)
- `reasoning_classified.csv`: 전체 데이터셋에 `reasoning_category` 컬럼이 부착된 상세 분석 데이터

---

## 3. Leave-one-verb-out (표본 불균형 강건성 검증)

이동동사(8개)처럼 표본이 작은 유형에서, 동사 하나를 뺐을 때 그룹 지표가 얼마나 흔들리는지 확인합니다.

```bash
# 이동동사만 (기본값)
python leave_one_verb_out.py results/main.jsonl \
  --group C --metric TBR_C --verb-classes Motion_to_Goal \
  --out results/lovo_motion_fold.csv \
  --out-summary results/lovo_motion_summary.csv

# 4개 1차 유형 전체를 나란히 비교 (표본 크기 효과 대조)
python leave_one_verb_out.py results/main.jsonl \
  --group C --metric TBR_C --verb-classes all \
  --out results/lovo_all_fold.csv \
  --out-summary results/lovo_all_summary.csv

# 2차 라벨(문법범주) 기준 LOVO 검증
python leave_one_verb_out.py results/main.jsonl \
  --secondary-labels 비완결동사_이중라벨링_매핑.csv \
  --group C --metric TBR_C --verb-classes none --secondary-classes all \
  --out results/lovo_secondary_fold.csv \
  --out-summary results/lovo_secondary_summary.csv
```

요약표의 `fold_range`(최대-최소)와 `fold_std`가 클수록, 그 조건의 결론이 동사 하나에 좌우된다는 뜻입니다. `n_source_verbs` 컬럼(유형별 전체 동사 수 참고값: 이동 8 / 소비 9 / 창조 39 / 상태변화 44)과 나란히 보면 "표본이 작아서 흔들리는 것"인지 판단하기 쉽습니다. `--group A --metric ACC`로 그룹 A쪽도 동일하게 볼 수 있습니다.

---

## 4. 부트스트랩 신뢰구간 (동사 단위 클러스터 부트스트랩)

문항이 아니라 **동사**를 리샘플링 단위로 삼습니다(같은 동사에서 나온 문항은 독립이 아니므로). LOVO가 "동사 하나를 뺐을 때"의 결정론적 스냅샷이라면, 부트스트랩은 "동사 구성이 달라졌다면 있었을 값의 분포"를 보여줘 서로 보완적입니다.

```bash
# 1차 라벨 기준 부트스트랩
python bootstrap_ci.py results/main.jsonl --n-boot 10000 \
  --out results/bootstrap_summary.csv

# 1차 및 2차 라벨(문법범주) 포함 부트스트랩
python bootstrap_ci.py results/main.jsonl --n-boot 10000 \
  --secondary-labels 비완결동사_이중라벨링_매핑.csv \
  --out results/bootstrap_summary_with_secondary.csv
```

- 모든 (model, strategy) 조합에 대해 ACC_A/B/D, TBR_C(전체+1차/2차 세부유형별), FBR_C, Δ_AA(전체+1차/2차 세부유형별)의 95% percentile CI를 계산합니다.
- `Δ_AA`는 그룹 D와 그룹 C가 서로 다른 동사 풀이므로, 두 풀을 각각 독립적으로 리샘플링한 뒤 같은 반복 회차에서 차이를 계산해 결합 CI를 만듭니다.
- `--secondary-labels`를 지정하면 2차 라벨(secondary_class)에 대한 CI도 자동으로 함께 계산됩니다.
- `n_boot=10000` 기준 전체 실행은 데이터 규모에 따라 수 분 걸릴 수 있습니다. 빠른 확인용으로는 `--n-boot 1000` 정도로 줄이고, 최종 결과에는 10000 이상을 권장합니다.
- 동사 수가 2개 미만인 조건은 부트스트랩이 불가능해 `note` 컬럼에 사유가 남고 CI는 NaN으로 표시됩니다.

---

## 5. 파이프라인 전체 모듈을 함께 보는 추천 워크플로우

1. `metrics.py`로 전체 그림(ACC/TBR/FBR/Δ_AA, 1차 및 2차 세부 유형 간 격차) 및 **[Step 1]** 대조 표(`--out-contrast`)를 먼저 본다.
2. `analyze_reasoning.py`로 **[Step 2]** 사고 과정 오류 정량 분포를 도출하여, Qwen 등의 오판 원인(비완결동사 Telic 기준 오적용 비율, 중단사건 과해석 비율 등)을 정량적으로 증명한다.
3. `semantic_contrast()` 또는 `run_contrast_analysis()`에서 흥미로운 의미론적 격차(동일 완성동사 내 이동 vs 소비 100%p 차이 등)를 추린다.
4. 그 조건에 대해 `leave_one_verb_out.py`로 특정 동사 하나가 결과를 좌우하는지 확인한다.
5. `bootstrap_ci.py`로 신뢰구간을 구해, 좁은 표본(이동동사 등)에서 나온 차이가 통계적으로도 안정적인지(구간이 겹치지 않는지) 최종 확인한다.

---

## 6. 참고 — pilot.jsonl로 동작 확인

Notion 노트에 첨부된 `pilot.jsonl`(20문항 × 2모델 × 4전략 = 160행, 유형별 동사 1~2개뿐인 파일럿 데이터)로 세 스크립트가 모두 정상 동작하는 것을 확인했습니다. 파일럿은 유형별 동사 수가 적어(부트스트랩 최소 조건인 2개 미만인 경우 포함) 신뢰구간이 매우 넓거나 NaN으로 나올 수 있는데, 이는 스크립트 오류가 아니라 표본 크기가 실제로 작다는 정상적인 신호입니다. 본실행(400문항) 데이터로 돌리면 훨씬 안정적인 값이 나올 것입니다.

