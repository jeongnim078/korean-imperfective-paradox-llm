# -*- coding: utf-8 -*-
"""
data_utils.py
=============
run_experiment.py가 생성한 results/*.jsonl(또는 pilot.jsonl)을 읽어
metrics.py / leave_one_verb_out.py / bootstrap_ci.py가 공통으로 쓰는
DataFrame 형태로 변환하는 유틸 모듈.

README.md의 결과 스키마를 그대로 가정한다:
    id, group, verb_class, verb, premise, hypothesis, gold_label,
    model, model_name, strategy, predicted_label, parse_ok,
    raw_response, reasoning, possible_interruptions

그룹 이름은 "A_Interrupted_Accomplishment" 처럼 접두 알파벳 + 설명으로
되어 있는데, 이 모듈에서는 앞 글자만 뽑아 group_code(A/B/C/D)를 만들어
이후 스크립트들이 짧게 참조할 수 있게 한다.

Gold label 규칙 (파일럿 데이터로 확인됨):
    A (Interrupted_Accomplishment) -> gold "False"   (완결동사, 진행형이 완료를 함의하지 않음)
    B (Interrupted_Activity)       -> gold "True"    (비완결동사, 중단돼도 발생 함의)
    C (Ambiguous_Accomplishment)   -> gold "Unknown" (완결동사, 완료 여부 불확실)
    D (Ambiguous_Activity)         -> gold "True"    (비완결동사, 모호한 상황에서도 발생은 참)

이중 라벨링 체계 (2차 라벨 추가)
--------------------------------
연구 설계상 완결동사(그룹 A/C)에는 두 층위의 라벨이 함께 붙는다.
    1차 라벨 (verb_class)     : 원 논문과 비교하기 위한 4분류
                                 Motion_to_Goal / Change_of_State / Creation / Consumption
    2차 라벨 (secondary_class): 한국어 문법 테스트로 확인한 실제 어휘상 범주
                                 완성동사 / 정도달성동사 / 전형적 달성동사

두 라벨은 별도 CSV(예: 비완결동사_이중라벨링_매핑.csv)로 관리되며,
`동사` 컬럼을 키로 결과 DataFrame과 조인한다. 1차 라벨과 2차 라벨이
어긋나는 지점(예: 상위유형이 '창조/소비'처럼 복합 표기인 동사)은
"문법 범주는 같지만 의미가 다른 동사 간 비교" 축의 후보이므로 별도
플래그(primary_class_conflict)로 남겨두고 조용히 무시하지 않는다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# 그룹 코드 <-> 원본 group 문자열 접두 매핑
GROUP_CODES = ("A", "B", "C", "D")

# 완결(telic) 동사 세부 유형(1차 라벨). group A/C에서만 등장한다.
TELIC_VERB_CLASSES = ("Motion_to_Goal", "Change_of_State", "Creation", "Consumption")

# 원 논문/README 기준 세부 유형별 동사 수(불균형의 근거) — 참고용 상수
VERB_CLASS_SIZE_HINT = {
    "Motion_to_Goal": 8,
    "Consumption": 9,
    "Creation": 39,
    "Change_of_State": 44,
}

# 2차 라벨(어휘상 문법범주). 한국어 문법 테스트 기준으로 매긴 실제 범주.
SECONDARY_VERB_CLASSES = ("완성동사", "정도달성동사", "전형적 달성동사")


def normalize_secondary_class(raw: str) -> str:
    """'전형적_달성동사' 또는 '전형적달성동사' -> '전형적 달성동사'"""
    if not isinstance(raw, str) or not raw.strip():
        return raw
    clean = raw.replace("_", " ").strip()
    for valid in SECONDARY_VERB_CLASSES:
        if clean.replace(" ", "") == valid.replace(" ", ""):
            return valid
    return clean


LABELS = ("True", "False", "Unknown")

# 2차 라벨링 매핑 CSV의 '상위유형' 원문 표기 -> 1차 라벨(TELIC_VERB_CLASSES) 정규화 테이블.
# 괄호로 세부조건이 붙은 표기(예: '창조(요리)')는 base 문자열만 사용하고,
# '/'로 이어진 복합 표기(예: '창조/소비')는 해당 동사가 두 1차 범주에 걸쳐 있다는 뜻이다.
_PRIMARY_CLASS_BASE_MAP = {
    "상태변화": "Change_of_State",
    "창조": "Creation",
    "소비": "Consumption",
    "이동": "Motion_to_Goal",
}


def _group_code(group: str) -> str:
    """'A_Interrupted_Accomplishment' -> 'A'"""
    if not isinstance(group, str) or not group:
        return ""
    code = group.split("_", 1)[0].strip().upper()
    return code if code in GROUP_CODES else code


def _normalize_primary_labels(raw: str) -> list[str]:
    """'창조(요리)' -> ['Creation'] / '창조/소비' -> ['Creation', 'Consumption']"""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = raw.split("/")
    out = []
    for p in parts:
        base = re.sub(r"\(.*?\)", "", p).strip()
        mapped = _PRIMARY_CLASS_BASE_MAP.get(base)
        if mapped is None:
            raise ValueError(f"알 수 없는 상위유형 표기: '{p}' (원본: '{raw}')")
        out.append(mapped)
    return out


def load_results(
    path: str | Path,
    drop_unparsed: bool = True,
    secondary_labels_path: str | Path | None = None,
) -> pd.DataFrame:
    """results jsonl 파일을 읽어 DataFrame으로 변환.

    Parameters
    ----------
    path : jsonl 파일 경로 (한 줄에 한 호출 결과 JSON)
    drop_unparsed : True면 parse_ok == False 인 행을 지표 계산 대상에서
        제외한다(단, 제외되었다는 사실은 남기기 위해 'included' 컬럼을 둔다).
        원본 raw_response는 보존되므로 언제든 다시 검토 가능.
    secondary_labels_path : 이중 라벨링 매핑 CSV 경로. 주어지면 verb 기준으로
        조인해 secondary_class 등 컬럼을 추가한다. 주지 않으면 secondary_class는
        전부 NaN인 채로 남아, 2차 라벨 없이도 기존 스크립트가 그대로 동작한다.

    Returns
    -------
    pd.DataFrame with an extra 'group_code' (A/B/C/D) and 'included' column,
    그리고 secondary_labels_path를 줬다면 secondary_class 등 2차 라벨 컬럼.
    """
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"{path}: 읽은 결과가 0건입니다.")

    df = pd.DataFrame(rows)

    required = {"id", "group", "verb_class", "verb", "gold_label", "model",
                "strategy", "predicted_label", "parse_ok"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 필수 컬럼 누락 -> {sorted(missing)}")

    df["group_code"] = df["group"].map(_group_code)

    unknown_groups = sorted(set(df["group_code"]) - set(GROUP_CODES))
    if unknown_groups:
        raise ValueError(f"{path}: 알 수 없는 group 값 발견 -> {unknown_groups}")

    df["parse_ok"] = df["parse_ok"].astype(bool)
    if drop_unparsed:
        df["included"] = df["parse_ok"]
    else:
        df["included"] = True

    n_excluded = int((~df["included"]).sum())
    if n_excluded:
        print(f"[data_utils] parse_ok=False {n_excluded}건은 지표 계산에서 제외됩니다 "
              f"(drop_unparsed={drop_unparsed}). raw_response를 확인해 파서를 보완하세요.")

    secondary_cols = ("secondary_class", "primary_class_raw", "primary_class_candidates",
                       "primary_class_conflict", "primary_class_mapped",
                       "context_dependent", "note")

    if secondary_labels_path is not None:
        secondary_labels = load_secondary_labels(secondary_labels_path)
        df = attach_secondary_labels(df, secondary_labels)
    else:
        for col in secondary_cols:
            if col not in df.columns:
                df[col] = pd.NA

    return df


def load_secondary_labels(path: str | Path) -> pd.DataFrame:
    """2차(어휘상 문법범주) 라벨링 매핑 CSV를 읽어 정리된 DataFrame으로 변환.

    입력 CSV 컬럼: 번호, 동사, 상위유형, 하위조건, 한국어분류, 문맥의존여부, 비고

    출력 컬럼
    ---------
    verb                    : 동사 (조인 키)
    secondary_class         : 2차 라벨 (한국어분류) — 완성동사/정도달성동사/전형적 달성동사
    primary_class_raw       : 상위유형 원문 표기 (예: '창조(요리)', '창조/소비')
    primary_class_candidates: 원문을 정규화한 1차 라벨 리스트. 단일이면 원소 1개,
                               복합 표기(예: '창조/소비')면 원소 2개
    primary_class_conflict  : bool. 복합 표기라서 1차 라벨이 하나로 안 정해지면 True
    primary_class_mapped    : 단일로 정해질 때만 1차 라벨 문자열, 아니면 NaN
    context_dependent       : bool. 문맥의존여부 == 'Y'
    note                    : 비고
    """
    path = Path(path)
    raw = pd.read_csv(path)

    required = {"동사", "상위유형", "한국어분류"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: 필수 컬럼 누락 -> {sorted(missing)}")

    if raw["동사"].duplicated().any():
        dups = raw.loc[raw["동사"].duplicated(keep=False), "동사"].unique().tolist()
        raise ValueError(f"{path}: 동사 컬럼에 중복 존재 -> {dups}")

    if not raw["한국어분류"].isin(SECONDARY_VERB_CLASSES).all():
        bad = sorted(set(raw["한국어분류"]) - set(SECONDARY_VERB_CLASSES))
        raise ValueError(f"{path}: 알 수 없는 한국어분류 값 -> {bad} "
                          f"(허용값: {SECONDARY_VERB_CLASSES})")

    candidates = raw["상위유형"].map(_normalize_primary_labels)

    context_col = raw["문맥의존여부"] if "문맥의존여부" in raw.columns else pd.Series(["N"] * len(raw))
    note_col = raw["비고"] if "비고" in raw.columns else pd.Series([None] * len(raw))

    out = pd.DataFrame({
        "verb": raw["동사"],
        "secondary_class": raw["한국어분류"],
        "primary_class_raw": raw["상위유형"],
        "primary_class_candidates": candidates,
        "primary_class_conflict": candidates.map(lambda c: len(set(c)) > 1),
        "primary_class_mapped": candidates.map(lambda c: c[0] if len(set(c)) == 1 else pd.NA),
        "context_dependent": context_col.fillna("N").eq("Y"),
        "note": note_col,
    })
    return out


def attach_secondary_labels(
    df: pd.DataFrame,
    secondary_labels: pd.DataFrame,
    warn_on_mismatch: bool = True,
) -> pd.DataFrame:
    """load_results()로 만든 df에 2차 라벨을 verb 기준으로 조인.

    - 그룹 B/D(비완결/활동동사)는 2차 라벨링 대상이 아니므로 secondary_class가
      NaN으로 남는 것이 정상이다.
    - 그룹 A/C(완결동사)인데 매핑 테이블에 없는 동사가 있으면 2차 라벨링 누락으로
      보고 에러를 낸다(조용히 NaN으로 흘려보내면 이후 verb_class별 부트스트랩
      표본 수가 슬쩍 줄어드는 걸 놓치기 쉽다).
    - 1차 라벨(verb_class, 실험에서 실제로 쓰인 값)과 매핑 테이블의 상위유형이
      다르면(복합 표기가 아닌 경우에 한해) 경고만 출력하고 원본 verb_class를
      그대로 유지한다 — 실험 시점의 1차 분류가 논문 비교 기준이므로 덮어쓰지 않음.
    """
    merged = df.merge(secondary_labels, on="verb", how="left", validate="many_to_one")

    telic_mask = merged["group_code"].isin(("A", "C"))
    unmapped = merged.loc[telic_mask & merged["secondary_class"].isna(), "verb"].unique()
    if len(unmapped):
        raise ValueError(f"2차 라벨링 매핑에 없는 완결동사 발견 -> {sorted(unmapped)}")

    if warn_on_mismatch:
        conflict_free = telic_mask & merged["primary_class_conflict"].eq(False)
        mismatched = merged.loc[
            conflict_free & (merged["verb_class"] != merged["primary_class_mapped"]), "verb"
        ].unique()
        if len(mismatched):
            print(f"[data_utils] 주의: 1차 라벨(verb_class)과 2차 라벨링 매핑의 상위유형이 "
                  f"어긋나는 동사 -> {sorted(mismatched)}. 원본 verb_class를 그대로 사용합니다.")

    conflict_verbs = sorted(merged.loc[telic_mask & merged["primary_class_conflict"].eq(True), "verb"].unique())
    if conflict_verbs:
        print(f"[data_utils] 안내: 상위유형이 복합 표기(예: '창조/소비')인 동사 {len(conflict_verbs)}개 "
              f"-> {conflict_verbs}. 1차 라벨은 실험 시점 verb_class를 그대로 쓰되, "
              f"'문법 범주는 같지만 의미가 다른 동사 간 비교' 축에서 별도 검토 대상입니다.")

    return merged


def filter_df(
    df: pd.DataFrame,
    group_code: Optional[str] = None,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
    verb_class: Optional[str | Iterable[str]] = None,
    secondary_class: Optional[str | Iterable[str]] = None,
    only_included: bool = True,
) -> pd.DataFrame:
    """조건에 맞춰 DataFrame을 필터링하는 공통 헬퍼.

    verb_class(1차 라벨)와 secondary_class(2차 라벨)는 서로 독립적인 필터로,
    둘 다 지정하면 AND 조건으로 적용된다. 예: verb_class='Creation',
    secondary_class='완성동사' -> "창조 동사 중 실제로 완성동사인 것만".
    """
    out = df
    if only_included:
        out = out[out["included"]]
    if group_code is not None:
        out = out[out["group_code"] == group_code]
    if model is not None:
        out = out[out["model"] == model]
    if strategy is not None:
        out = out[out["strategy"] == strategy]
    if verb_class is not None:
        if isinstance(verb_class, str):
            out = out[out["verb_class"] == verb_class]
        else:
            out = out[out["verb_class"].isin(list(verb_class))]
    if secondary_class is not None:
        if isinstance(secondary_class, str):
            out = out[out["secondary_class"] == secondary_class]
        else:
            out = out[out["secondary_class"].isin(list(secondary_class))]
    return out


def model_strategy_combos(df: pd.DataFrame) -> list[tuple[str, str]]:
    """데이터에 실제로 존재하는 (model, strategy) 조합 목록."""
    return sorted(set(map(tuple, df[["model", "strategy"]].drop_duplicates().values.tolist())))
