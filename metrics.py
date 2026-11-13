# -*- coding: utf-8 -*-
"""
metrics.py
==========
원 논문(Ma & Miyao, 2026)의 정의를 그대로 한국어 데이터셋에 적용한
평가지표를 계산한다.

(1) Group-wise Accuracy (ACC)
    각 논리 조건(그룹 A-D)별 정확도.
    ACC_g = (1/|g|) * sum_{i in g} 1[y_hat_i == gold_i]

(2) Teleological Bias Rate (TBR_C)
    그룹 C(모호한 완결, gold=Unknown)에서 모델이 "참"이라고 답한 비율.
    완료를 환각(hallucinate)하는 정도를 직접 측정한다.
    TBR_C = (1/|C|) * sum_{i in C} 1[y_hat_i == "True"]

(3) False Belief Rate (FBR_C)  [추가]
    그룹 C에서 모델이 "거짓"이라고 답한 비율 — TBR_C와 대칭인 지표로,
    "완료를 환각"하는 게 아니라 반대로 "미완료를 과신"하는 정도를 측정한다.
    FBR_C = (1/|C|) * sum_{i in C} 1[y_hat_i == "False"]
    ACC_C = 1 - TBR_C - FBR_C - (파싱실패 등 기타 비율) 관계이므로, ACC_C가
    낮게 나온 조건에서 그 손실이 TBR_C(완결 환각)와 FBR_C(미완료 과신) 중
    어느 쪽에서 온 것인지 분해해서 볼 수 있다.

(4) Δ_AA (Accomplishment-Activity gap)
    Δ_AA = ACC_D - TBR_C
    그룹 D(모호한 비완결, gold=True)에서의 정답률 대비, 그룹 C에서
    "완료"를 환각하는 비율이 얼마나 큰지를 대조한다. 값이 클수록
    비완결/완결 구분 능력과 목적론적 편향 사이의 괴리가 크다는 뜻.

이중 라벨링 (2차 라벨) 지원
----------------------------
완결동사(그룹 A/C)에는 1차 라벨(verb_class: 원 논문 4분류)과 2차 라벨
(secondary_class: 한국어 문법 테스트로 확인한 실제 어휘상 범주 — 완성동사/
정도달성동사/전형적 달성동사)이 함께 붙는다. data_utils.load_results에
secondary_labels_path를 주면 df에 secondary_class 컬럼이 채워지고,
이 모듈의 ACC/TBR_C/FBR_C/delta_aa는 모두 secondary_class 인자를 받아
1차·2차 라벨을 독립적으로(또는 동시에) 필터링할 수 있다.

이 모듈의 모든 함수는 (model, strategy, verb_class, secondary_class) 단위로
세분화해서 계산할 수 있으며, leave_one_verb_out.py / bootstrap_ci.py에서
동일한 함수를 재사용한다(계산 로직이 여러 곳에서 어긋나지 않도록 하기 위함).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data_utils import (
    SECONDARY_VERB_CLASSES,
    TELIC_VERB_CLASSES,
    filter_df,
    load_results,
    model_strategy_combos,
)


# ---------------------------------------------------------------------------
# 기본 지표 함수 (df를 직접 받는 버전 -- leave-one-out / bootstrap에서 재사용)
# ---------------------------------------------------------------------------

def acc_on(df: pd.DataFrame) -> tuple[float, int]:
    """이미 group/model/strategy/verb_class로 필터링된 df에 대한 accuracy.

    Returns (accuracy, n). n == 0이면 accuracy는 nan.
    """
    n = len(df)
    if n == 0:
        return float("nan"), 0
    correct = (df["predicted_label"] == df["gold_label"]).sum()
    return correct / n, n


def tbr_on(df: pd.DataFrame) -> tuple[float, int]:
    """이미 group C로 필터링된 df에 대한 TBR (predicted_label == 'True' 비율)."""
    n = len(df)
    if n == 0:
        return float("nan"), 0
    hallucinated = (df["predicted_label"] == "True").sum()
    return hallucinated / n, n


def fbr_on(df: pd.DataFrame) -> tuple[float, int]:
    """이미 group C로 필터링된 df에 대한 FBR (predicted_label == 'False' 비율).

    TBR_C(완결 환각)와 대칭인 지표로, "모호한데 거짓이라고 과잉 단정"하는
    정도 -- 미완료 과신(false-belief bias) -- 를 측정한다.
    """
    n = len(df)
    if n == 0:
        return float("nan"), 0
    false_biased = (df["predicted_label"] == "False").sum()
    return false_biased / n, n


# ---------------------------------------------------------------------------
# 조건(model/strategy/verb_class/secondary_class) 기반 편의 함수
# ---------------------------------------------------------------------------

def ACC(
    df: pd.DataFrame,
    group_code: str,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
    verb_class: Optional[str | Iterable[str]] = None,
    secondary_class: Optional[str | Iterable[str]] = None,
) -> tuple[float, int]:
    """그룹 group_code('A'|'B'|'C'|'D')의 정확도. verb_class(1차)/secondary_class(2차)로
    세부 유형 제한 가능(둘 다 주면 AND 조건)."""
    sub = filter_df(df, group_code=group_code, model=model, strategy=strategy,
                     verb_class=verb_class, secondary_class=secondary_class)
    return acc_on(sub)


def TBR_C(
    df: pd.DataFrame,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
    verb_class: Optional[str | Iterable[str]] = None,
    secondary_class: Optional[str | Iterable[str]] = None,
) -> tuple[float, int]:
    """그룹 C 전용 목적론적 편향률(완결 환각)."""
    sub = filter_df(df, group_code="C", model=model, strategy=strategy,
                     verb_class=verb_class, secondary_class=secondary_class)
    return tbr_on(sub)


def FBR_C(
    df: pd.DataFrame,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
    verb_class: Optional[str | Iterable[str]] = None,
    secondary_class: Optional[str | Iterable[str]] = None,
) -> tuple[float, int]:
    """그룹 C 전용 미완료 과신률(TBR_C와 대칭)."""
    sub = filter_df(df, group_code="C", model=model, strategy=strategy,
                     verb_class=verb_class, secondary_class=secondary_class)
    return fbr_on(sub)


def delta_aa(
    df: pd.DataFrame,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
    verb_class: Optional[str | Iterable[str]] = None,
    secondary_class: Optional[str | Iterable[str]] = None,
) -> tuple[float, int, int]:
    """Δ_AA = ACC_D - TBR_C.

    ACC_D는 그룹 D(Activity, 세부 유형 없음)에서 계산하므로 verb_class /
    secondary_class 인자의 영향을 받지 않는다. 두 인자는 TBR_C 쪽 완결동사
    세부 유형(1차/2차)만 제한한다.

    Returns (delta, n_D, n_C)
    """
    acc_d, n_d = ACC(df, "D", model=model, strategy=strategy, verb_class=None, secondary_class=None)
    tbr_c, n_c = TBR_C(df, model=model, strategy=strategy, verb_class=verb_class, secondary_class=secondary_class)
    return acc_d - tbr_c, n_d, n_c


# ---------------------------------------------------------------------------
# 요약 테이블
# ---------------------------------------------------------------------------

def _filtered_combos(df, models, strategies):
    combos = model_strategy_combos(df)
    if models is not None:
        models = set(models)
        combos = [(m, s) for m, s in combos if m in models]
    if strategies is not None:
        strategies = set(strategies)
        combos = [(m, s) for m, s in combos if s in strategies]
    return combos


def summarize(
    df: pd.DataFrame,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """(model, strategy) 조합별 ACC_A/B/C/D, TBR_C, FBR_C, Δ_AA 요약표.

    models/strategies를 주면 해당 값들만 남겨 계산한다(예: 아직 결과가
    불완전한 모델을 이번 요약에서 잠정 제외하고 싶을 때).
    """
    rows = []
    combos = _filtered_combos(df, models, strategies)
    for model, strategy in combos:
        acc_a, n_a = ACC(df, "A", model, strategy)
        acc_b, n_b = ACC(df, "B", model, strategy)
        acc_c, n_c = ACC(df, "C", model, strategy)  # 참고용(정답=Unknown 기준 accuracy)
        acc_d, n_d = ACC(df, "D", model, strategy)
        tbr_c, _ = TBR_C(df, model, strategy)
        fbr_c, _ = FBR_C(df, model, strategy)
        delta, _, _ = delta_aa(df, model, strategy)
        rows.append({
            "model": model, "strategy": strategy,
            "ACC_A": acc_a, "n_A": n_a,
            "ACC_B": acc_b, "n_B": n_b,
            "ACC_C": acc_c, "n_C": n_c,
            "ACC_D": acc_d, "n_D": n_d,
            "TBR_C": tbr_c,
            "FBR_C": fbr_c,
            "Delta_AA": delta,
        })
    return pd.DataFrame(rows).sort_values(["model", "strategy"]).reset_index(drop=True)


def summarize_by_verb_class(
    df: pd.DataFrame,
    groups: Iterable[str] = ("A", "C"),
    verb_classes: Iterable[str] = TELIC_VERB_CLASSES,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """(model, strategy, verb_class) 단위 -- 1차 라벨 기준 -- 그룹 A/C 세부 유형별 지표 요약.

    그룹 A는 ACC를, 그룹 C는 ACC/TBR_C/FBR_C를 함께 낸다(둘 다 verb_class로
    제한된 부분집합 기준). 원 논문과의 비교(4분류) 축에 그대로 쓸 수 있는 표.
    """
    rows = []
    combos = _filtered_combos(df, models, strategies)
    for model, strategy in combos:
        for vc in verb_classes:
            row = {"model": model, "strategy": strategy, "verb_class": vc}
            if "A" in groups:
                acc_a, n_a = ACC(df, "A", model, strategy, verb_class=vc)
                row["ACC_A"] = acc_a
                row["n_A"] = n_a
            if "C" in groups:
                acc_c, n_c = ACC(df, "C", model, strategy, verb_class=vc)
                tbr_c, _ = TBR_C(df, model, strategy, verb_class=vc)
                fbr_c, _ = FBR_C(df, model, strategy, verb_class=vc)
                row["ACC_C"] = acc_c
                row["TBR_C"] = tbr_c
                row["FBR_C"] = fbr_c
                row["n_C"] = n_c
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "strategy", "verb_class"]).reset_index(drop=True)


def summarize_by_secondary_class(
    df: pd.DataFrame,
    groups: Iterable[str] = ("A", "C"),
    secondary_classes: Iterable[str] = SECONDARY_VERB_CLASSES,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """(model, strategy, secondary_class) 단위 -- 2차 라벨(한국어 문법범주) 기준 요약.

    "이동 vs 상태변화(혹은 이동/상태변화 vs 완성동사)"처럼 문법 범주 자체가
    다른 조건을 비교하는 축. secondary_labels_path 없이 load_results를
    호출했다면 secondary_class가 전부 NaN이라 결과가 비어 있을 수 있다.
    """
    rows = []
    combos = _filtered_combos(df, models, strategies)
    for model, strategy in combos:
        for sc in secondary_classes:
            row = {"model": model, "strategy": strategy, "secondary_class": sc}
            if "A" in groups:
                acc_a, n_a = ACC(df, "A", model, strategy, secondary_class=sc)
                row["ACC_A"] = acc_a
                row["n_A"] = n_a
            if "C" in groups:
                acc_c, n_c = ACC(df, "C", model, strategy, secondary_class=sc)
                tbr_c, _ = TBR_C(df, model, strategy, secondary_class=sc)
                fbr_c, _ = FBR_C(df, model, strategy, secondary_class=sc)
                row["ACC_C"] = acc_c
                row["TBR_C"] = tbr_c
                row["FBR_C"] = fbr_c
                row["n_C"] = n_c
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "strategy", "secondary_class"]).reset_index(drop=True)


def semantic_contrast(
    df: pd.DataFrame,
    group_code: str,
    primary_classes: tuple[str, str],
    secondary_class: str,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """같은 2차 라벨(문법범주) 안에서 서로 다른 1차 라벨(어휘 의미) 간 지표를
    나란히 비교 -- "문법 범주는 같지만 의미가 다른 동사 간 비교" 축 전용 헬퍼.

    예: semantic_contrast(df, "C", ("Creation", "Consumption"), "완성동사")
        -> 완성동사로 확인된 창조/소비 동사만 남겨 ACC_C/TBR_C/FBR_C를 비교.
           (전형적달성동사로 재분류된 소비 동사 1개 등은 자동으로 제외되어
           "동일 문법범주 내 순수 의미론적 대조"가 유지된다.)

    group_code='A'면 ACC_A만, 'C'면 ACC_C/TBR_C/FBR_C까지 함께 낸다.
    """
    rows = []
    combos = _filtered_combos(df, models, strategies)
    for model, strategy in combos:
        for pc in primary_classes:
            row = {"model": model, "strategy": strategy,
                   "verb_class": pc, "secondary_class": secondary_class}
            acc, n = ACC(df, group_code, model, strategy, verb_class=pc, secondary_class=secondary_class)
            row[f"ACC_{group_code}"] = acc
            row["n"] = n
            if group_code == "C":
                tbr, _ = TBR_C(df, model, strategy, verb_class=pc, secondary_class=secondary_class)
                fbr, _ = FBR_C(df, model, strategy, verb_class=pc, secondary_class=secondary_class)
                row["TBR_C"] = tbr
                row["FBR_C"] = fbr
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "strategy", "verb_class"]).reset_index(drop=True)


DEFAULT_CONTRAST_SPECS = [
    {
        "contrast_name": "이동(완성) vs 소비(완성)",
        "contrast_type": "동일 문법 범주 내 순수 의미 차이",
        "side_A": {"verb_class": "Motion_to_Goal", "secondary_class": "완성동사", "label": "이동(완성동사)"},
        "side_B": {"verb_class": "Consumption", "secondary_class": "완성동사", "label": "소비(완성동사)"},
    },
    {
        "contrast_name": "이동(달성) vs 소비(완성)",
        "contrast_type": "서로 다른 문법 범주 간 차이",
        "side_A": {"verb_class": "Motion_to_Goal", "secondary_class": "전형적 달성동사", "label": "이동(전형적 달성동사)"},
        "side_B": {"verb_class": "Consumption", "secondary_class": "완성동사", "label": "소비(완성동사)"},
    },
    {
        "contrast_name": "창조(완성) vs 소비(완성)",
        "contrast_type": "동일 문법 범주 내 순수 의미 차이",
        "side_A": {"verb_class": "Creation", "secondary_class": "완성동사", "label": "창조(완성동사)"},
        "side_B": {"verb_class": "Consumption", "secondary_class": "완성동사", "label": "소비(완성동사)"},
    },
]


def run_contrast_analysis(
    df: pd.DataFrame,
    group_code: str = "C",
    specs: Optional[list[dict]] = None,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Step 1 전용 헬퍼: 1차/2차 이중 라벨 조합 간 TBR_C, FBR_C, ACC 대조 분석.

    동일 문법 범주 내 순수 의미론적 차이(예: 이동 완성 vs 소비 완성) 및
    서로 다른 문법 범주 간 차이(예: 이동 전형적달성 vs 소비 완성)를
    (model, strategy)별로 비교하여 퍼센트포인트 격차(TBR_C_diff_pp, FBR_C_diff_pp)를 산출한다.
    """
    if specs is None:
        specs = DEFAULT_CONTRAST_SPECS

    rows = []
    combos = _filtered_combos(df, models, strategies)
    for model, strategy in combos:
        for spec in specs:
            s_a = spec["side_A"]
            s_b = spec["side_B"]

            tbr_a, n_a = TBR_C(df, model, strategy, verb_class=s_a.get("verb_class"), secondary_class=s_a.get("secondary_class"))
            fbr_a, _ = FBR_C(df, model, strategy, verb_class=s_a.get("verb_class"), secondary_class=s_a.get("secondary_class"))
            acc_a, _ = ACC(df, group_code, model, strategy, verb_class=s_a.get("verb_class"), secondary_class=s_a.get("secondary_class"))

            tbr_b, n_b = TBR_C(df, model, strategy, verb_class=s_b.get("verb_class"), secondary_class=s_b.get("secondary_class"))
            fbr_b, _ = FBR_C(df, model, strategy, verb_class=s_b.get("verb_class"), secondary_class=s_b.get("secondary_class"))
            acc_b, _ = ACC(df, group_code, model, strategy, verb_class=s_b.get("verb_class"), secondary_class=s_b.get("secondary_class"))

            tbr_diff = (tbr_a - tbr_b) * 100 if not np.isnan(tbr_a) and not np.isnan(tbr_b) else float("nan")
            fbr_diff = (fbr_a - fbr_b) * 100 if not np.isnan(fbr_a) and not np.isnan(fbr_b) else float("nan")

            rows.append({
                "model": model,
                "strategy": strategy,
                "contrast_name": spec["contrast_name"],
                "contrast_type": spec["contrast_type"],
                "side_A": s_a.get("label", s_a.get("verb_class")),
                "n_A": n_a,
                "TBR_C_A": tbr_a,
                "FBR_C_A": fbr_a,
                "ACC_C_A": acc_a,
                "side_B": s_b.get("label", s_b.get("verb_class")),
                "n_B": n_b,
                "TBR_C_B": tbr_b,
                "FBR_C_B": fbr_b,
                "ACC_C_B": acc_b,
                "TBR_C_diff_pp": tbr_diff,
                "FBR_C_diff_pp": fbr_diff,
            })

    return pd.DataFrame(rows).sort_values(["contrast_name", "model", "strategy"]).reset_index(drop=True)



def class_gap(
    df: pd.DataFrame,
    group_code: str = "C",
    metric: str = "TBR_C",
    class_column: str = "verb_class",
    classes: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """(model, strategy)별로 세부 유형 간 지표 최대-최소 격차(퍼센트포인트).

    class_column='verb_class'면 1차 라벨(원 논문 4분류), 'secondary_class'면
    2차 라벨(한국어 문법범주) 기준으로 격차를 계산한다. 연구계획서에 적어둔
    사전 기준("세부 유형 간 정확도/편향 격차 threshold 이상")을 결과가 나온
    뒤 바로 확인할 수 있게 하는 헬퍼.
    """
    if class_column == "verb_class":
        classes = TELIC_VERB_CLASSES if classes is None else classes
        by_c = summarize_by_verb_class(df, groups=(group_code,), verb_classes=classes,
                                        models=models, strategies=strategies)
        class_col_name = "verb_class"
    elif class_column == "secondary_class":
        classes = SECONDARY_VERB_CLASSES if classes is None else classes
        by_c = summarize_by_secondary_class(df, groups=(group_code,), secondary_classes=classes,
                                             models=models, strategies=strategies)
        class_col_name = "secondary_class"
    else:
        raise ValueError("class_column은 'verb_class' 또는 'secondary_class'만 지원합니다.")

    if metric not in by_c.columns:
        raise ValueError(f"metric '{metric}'은 group_code='{group_code}'에 존재하지 않습니다. "
                          f"사용 가능: {[c for c in by_c.columns if c not in ('model', 'strategy', class_col_name)]}")

    rows = []
    for (model, strategy), g in by_c.groupby(["model", "strategy"]):
        vals = g[metric].dropna()
        if vals.empty:
            continue
        gap = vals.max() - vals.min()
        best = g.loc[vals.idxmax(), class_col_name]
        worst = g.loc[vals.idxmin(), class_col_name]
        rows.append({
            "model": model, "strategy": strategy, "metric": metric, "class_column": class_col_name,
            "gap_pp": gap * 100,
            f"max_{class_col_name}": best, "max_value": vals.max(),
            f"min_{class_col_name}": worst, "min_value": vals.min(),
            "meets_threshold": bool(gap >= threshold),
        })
    return pd.DataFrame(rows).sort_values("gap_pp", ascending=False).reset_index(drop=True)


def verb_class_gap(
    df: pd.DataFrame,
    group_code: str = "C",
    metric: str = "TBR_C",
    verb_classes: Iterable[str] = TELIC_VERB_CLASSES,
    models: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """기존 이름 유지용 얇은 래퍼(1차 라벨 기준 격차, 사전 기준 15%p)."""
    return class_gap(df, group_code=group_code, metric=metric, class_column="verb_class",
                      classes=verb_classes, models=models, strategies=strategies, threshold=0.15)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="ACC / TBR_C / FBR_C / Delta_AA 요약표 계산")
    ap.add_argument("results_path", help="run_experiment.py가 만든 jsonl 결과 파일 경로")
    ap.add_argument("--secondary-labels", default=None,
                     help="이중 라벨링 매핑 CSV 경로(동사, 상위유형, 한국어분류 등). "
                          "주면 2차 라벨(secondary_class) 기준 요약표도 함께 출력.")
    ap.add_argument("--out-summary", default=None, help="model x strategy 요약 CSV 저장 경로")
    ap.add_argument("--out-by-verb-class", default=None, help="1차 라벨(세부 유형)별 요약 CSV 저장 경로")
    ap.add_argument("--out-by-secondary-class", default=None, help="2차 라벨(문법범주)별 요약 CSV 저장 경로")
    ap.add_argument("--out-gap", default=None, help="1차 라벨 세부 유형 간 격차 CSV 저장 경로")
    ap.add_argument("--out-contrast", default=None, help="Step 1 대조 분석(이동 완성 vs 소비 완성 등) CSV 저장 경로")
    ap.add_argument("--models", nargs="+", default=None,
                     help="지정한 모델만 요약에 포함(예: --models gpt qwen). "
                          "기본값은 데이터에 있는 모든 모델.")
    ap.add_argument("--strategies", nargs="+", default=None,
                     help="지정한 전략만 요약에 포함. 기본값은 데이터에 있는 모든 전략.")
    ap.add_argument("--keep-unparsed", action="store_true",
                     help="parse_ok=False 행도 지표 계산에 포함(기본은 제외)")
    args = ap.parse_args()

    df = load_results(args.results_path, drop_unparsed=not args.keep_unparsed,
                       secondary_labels_path=args.secondary_labels)

    if args.models:
        present = sorted(df["model"].unique())
        missing = set(args.models) - set(present)
        if missing:
            raise SystemExit(f"--models에 지정한 값 중 데이터에 없는 모델: {sorted(missing)} "
                              f"(데이터에 있는 모델: {present})")
        print(f"[안내] --models {args.models} 로 필터링하여 계산합니다 "
              f"(데이터 전체 모델: {present}).")

    summary = summarize(df, models=args.models, strategies=args.strategies)
    pd.set_option("display.width", 140)
    print("\n=== (model, strategy)별 ACC/TBR_C/FBR_C/Delta_AA ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    by_vc = summarize_by_verb_class(df, models=args.models, strategies=args.strategies)
    print("\n=== 그룹 A/C 1차 라벨(세부 동사 유형)별 요약 ===")
    print(by_vc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    gap = verb_class_gap(df, group_code="C", metric="TBR_C", models=args.models, strategies=args.strategies)
    print("\n=== 그룹 C: 1차 라벨 세부 유형 간 TBR 격차 (사전 기준 15%p) ===")
    print(gap.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    by_sc = None
    contrast_df = None
    if args.secondary_labels:
        by_sc = summarize_by_secondary_class(df, models=args.models, strategies=args.strategies)
        print("\n=== 그룹 A/C 2차 라벨(한국어 문법범주)별 요약 ===")
        print(by_sc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        contrast_df = run_contrast_analysis(df, group_code="C", models=args.models, strategies=args.strategies)
        print("\n=== [Step 1] 그룹 C: 2차 라벨 세부 대조 분석 (TBR_C / FBR_C 격차) ===")
        print(contrast_df[["model", "strategy", "contrast_name", "side_A", "TBR_C_A", "side_B", "TBR_C_B", "TBR_C_diff_pp", "FBR_C_diff_pp"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.out_summary:
        summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")
        print(f"\n[저장됨] {args.out_summary}")
    if args.out_by_verb_class:
        by_vc.to_csv(args.out_by_verb_class, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_by_verb_class}")
    if args.out_by_secondary_class and by_sc is not None:
        by_sc.to_csv(args.out_by_secondary_class, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_by_secondary_class}")
    if args.out_gap:
        gap.to_csv(args.out_gap, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_gap}")
    if args.out_contrast and contrast_df is not None:
        contrast_df.to_csv(args.out_contrast, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_contrast}")



if __name__ == "__main__":
    _cli()
