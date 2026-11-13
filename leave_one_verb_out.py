# -*- coding: utf-8 -*-
"""
leave_one_verb_out.py
======================
소표본 동사 유형(특히 이동동사 Motion_to_Goal, n=8)에서 그룹 평균이
동사 하나에 의해 얼마나 흔들리는지를 검증하는 leave-one-verb-out(LOVO) 스크립트.

절차 (동사 유형 하나, 그룹 하나, (model,strategy) 조합 하나 기준):
    1. 해당 verb_class(1차) 또는 secondary_class(2차)에 속한 전체 동사 목록 V = {v_1, ..., v_k}를 구한다.
    2. 전체 동사를 사용한 baseline 지표를 계산한다.
    3. i = 1..k에 대해, v_i를 제외한 나머지 (k-1)개 동사만으로 같은 지표를
       재계산한다 -> LOVO fold 값 k개.
    4. fold 값들의 평균/표준편차/최대-최소 범위(range)를 baseline과 비교해,
       "특정 동사 하나가 그룹 평균을 얼마나 흔드는지"를 정량화한다.

그룹 A(완결/중단)에는 ACC를, 그룹 C(완결/모호)에는 ACC, TBR_C, FBR_C를 함께 적용한다.
(TBR_C가 원 논문이 정의한 핵심 편향 지표이므로 기본적으로 TBR_C를 우선 본다.)

이중 라벨링 (2차 라벨) 지원
----------------------------
완결동사(그룹 A/C)에는 1차 라벨(verb_class: 원 논문 4분류)과 2차 라벨
(secondary_class: 한국어 문법 테스트로 확인한 실제 어휘상 범주 — 완성동사/
정도달성동사/전형적 달성동사)이 함께 붙는다. CLI에 --secondary-labels 옵션으로
매핑 CSV를 전달하면 2차 라벨(--secondary-classes) 기준 LOVO도 함께 수행할 수 있다.

동사 수가 매우 작을 때(k=8인 이동동사 등)는 range/std가 크게 나오는 것
자체가 "이 그룹 크기에서는 결론을 신중히 봐야 한다"는 신호가 된다.
표본이 큰 그룹(예: 상태변화동사 44개)과 나란히 비교하면 표본 크기 효과를
직접 보여줄 수 있다.

사용 예:
    python leave_one_verb_out.py results/main.jsonl \\
        --group C --metric TBR_C --verb-classes Motion_to_Goal \\
        --out results/lovo_motion_tbr.csv

    # 4개 유형 전체를 한 번에 비교 (표본 크기 효과 대조)
    python leave_one_verb_out.py results/main.jsonl \\
        --group C --metric TBR_C --verb-classes all \\
        --out results/lovo_all_tbr.csv

    # 2차 라벨(문법범주) 기준 LOVO
    python leave_one_verb_out.py results/main.jsonl \\
        --secondary-labels 비완결동사_이중라벨링_매핑.csv \\
        --secondary-classes all \\
        --out results/lovo_secondary.csv
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from data_utils import (
    SECONDARY_VERB_CLASSES,
    TELIC_VERB_CLASSES,
    VERB_CLASS_SIZE_HINT,
    filter_df,
    load_results,
    model_strategy_combos,
    normalize_secondary_class,
)
from metrics import acc_on, fbr_on, tbr_on

MetricFn = Callable[[pd.DataFrame], tuple[float, int]]

METRIC_FUNCS: dict[str, MetricFn] = {
    "ACC": acc_on,
    "TBR_C": tbr_on,
    "FBR_C": fbr_on,
}


def _verbs_in_class(
    df: pd.DataFrame,
    group_code: str,
    model: str,
    strategy: str,
    verb_class: str | None = None,
    secondary_class: str | None = None,
) -> list[str]:
    sub = filter_df(df, group_code=group_code, model=model, strategy=strategy,
                    verb_class=verb_class, secondary_class=secondary_class)
    return sorted(sub["verb"].dropna().unique().tolist())


def lovo_for_condition(
    df: pd.DataFrame,
    group_code: str,
    model: str,
    strategy: str,
    verb_class: str | None = None,
    secondary_class: str | None = None,
    metric_name: str = "TBR_C",
) -> pd.DataFrame:
    """단일 (group, verb_class/secondary_class, model, strategy) 조합에 대한 LOVO fold 결과.

    반환 DataFrame 컬럼:
        group, verb_class, secondary_class, model, strategy,
        excluded_verb, n_verbs_used, n_items_used, metric_value,
        baseline_value, delta_from_baseline
    excluded_verb == '(baseline)'인 행이 전체 동사를 사용한 기준값이다.
    """
    metric_fn = METRIC_FUNCS[metric_name]
    base_sub = filter_df(
        df, group_code=group_code, model=model, strategy=strategy,
        verb_class=verb_class, secondary_class=secondary_class
    )
    verbs = sorted(base_sub["verb"].dropna().unique().tolist())

    if not verbs:
        return pd.DataFrame()

    baseline_val, baseline_n = metric_fn(base_sub)

    rows = [{
        "group": group_code, "verb_class": verb_class, "secondary_class": secondary_class,
        "model": model, "strategy": strategy,
        "excluded_verb": "(baseline)", "n_verbs_used": len(verbs), "n_items_used": baseline_n,
        "metric_value": baseline_val, "baseline_value": baseline_val,
        "delta_from_baseline": 0.0,
    }]

    for v in verbs:
        loo_sub = base_sub[base_sub["verb"] != v]
        val, n = metric_fn(loo_sub)
        rows.append({
            "group": group_code, "verb_class": verb_class, "secondary_class": secondary_class,
            "model": model, "strategy": strategy,
            "excluded_verb": v, "n_verbs_used": len(verbs) - 1, "n_items_used": n,
            "metric_value": val, "baseline_value": baseline_val,
            "delta_from_baseline": (val - baseline_val) if not (np.isnan(val) or np.isnan(baseline_val)) else np.nan,
        })

    return pd.DataFrame(rows)


def lovo_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    """lovo_for_condition()의 fold별 결과를 (group, verb_class/secondary_class, model, strategy) 단위로 집계.

    fold(=baseline 제외) 값들의 std, range(max-min), 최대 이탈 동사(가장 크게
    흔든 동사)를 보고한다. n_verbs가 작을수록 range/std가 커지는 경향을
    n_source_verbs(원래 유형 크기, 참고용)와 함께 확인할 수 있다.
    """
    rows = []
    possible_keys = ["group", "verb_class", "secondary_class", "model", "strategy"]
    keys = [k for k in possible_keys if k in fold_df.columns]
    if not keys:
        return pd.DataFrame()

    for key_vals, g in fold_df.groupby(keys, dropna=False):
        base = g[g["excluded_verb"] == "(baseline)"]
        folds = g[g["excluded_verb"] != "(baseline)"].dropna(subset=["metric_value"])
        if base.empty or folds.empty:
            continue
        baseline_val = base["metric_value"].iloc[0]
        vals = folds["metric_value"].to_numpy()
        max_row = folds.loc[folds["delta_from_baseline"].abs().idxmax()]

        if isinstance(key_vals, tuple):
            row = dict(zip(keys, key_vals))
        else:
            row = {keys[0]: key_vals}

        vc = row.get("verb_class")
        if pd.notna(vc) and vc in VERB_CLASS_SIZE_HINT:
            n_src = VERB_CLASS_SIZE_HINT[vc]
        else:
            n_src = base["n_verbs_used"].iloc[0]

        row.update({
            "baseline_value": baseline_val,
            "n_folds": len(folds),
            "fold_mean": float(vals.mean()),
            "fold_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "fold_range": float(vals.max() - vals.min()),
            "max_abs_delta": float(abs(max_row["delta_from_baseline"])),
            "most_influential_verb": max_row["excluded_verb"],
            "n_source_verbs": n_src,
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fold_range", ascending=False).reset_index(drop=True)


def run_lovo(
    df: pd.DataFrame,
    group_code: str,
    metric_name: str,
    verb_classes: list[str] | None = None,
    secondary_classes: list[str] | None = None,
    models: list[str] | None = None,
    strategies: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전체 (model,strategy) x verb_class / secondary_class 조합에 대해 LOVO를 실행.

    Returns (fold_level_df, summary_df)
    """
    combos = model_strategy_combos(df)
    if models is not None:
        combos = [(m, s) for m, s in combos if m in models]
    if strategies is not None:
        combos = [(m, s) for m, s in combos if s in strategies]

    all_folds = []
    if verb_classes:
        for vc in verb_classes:
            for model, strategy in combos:
                fold_df = lovo_for_condition(df, group_code, model, strategy, verb_class=vc, metric_name=metric_name)
                if not fold_df.empty:
                    all_folds.append(fold_df)

    if secondary_classes:
        for sc in secondary_classes:
            for model, strategy in combos:
                fold_df = lovo_for_condition(df, group_code, model, strategy, secondary_class=sc, metric_name=metric_name)
                if not fold_df.empty:
                    all_folds.append(fold_df)

    if not all_folds:
        raise ValueError("LOVO 결과가 비었습니다. group/verb_class/secondary_class/model/strategy 조합을 확인하세요.")

    fold_level = pd.concat(all_folds, ignore_index=True)
    summary = lovo_summary(fold_level)
    return fold_level, summary


def _cli():
    ap = argparse.ArgumentParser(description="Leave-one-verb-out 강건성 검증")
    ap.add_argument("results_path")
    ap.add_argument("--secondary-labels", default=None,
                     help="이중 라벨링 매핑 CSV 경로. 주면 2차 라벨(secondary_class) 기준 LOVO도 함께 수행 가능.")
    ap.add_argument("--group", choices=["A", "C"], default="C",
                     help="LOVO를 적용할 그룹 (완결동사 세부 유형은 A/C에만 존재)")
    ap.add_argument("--metric", choices=["ACC", "TBR_C", "FBR_C"], default="TBR_C")
    ap.add_argument("--verb-classes", nargs="+", default=["Motion_to_Goal"],
                     help="'all'을 주면 4개 완결 세부 유형 전체 비교. 기본값은 이동동사만. 생략하려면 'none'.")
    ap.add_argument("--secondary-classes", nargs="+", default=None,
                     help="'all'을 주면 3개 문법범주(완성동사/정도달성동사/전형적 달성동사) 전체 비교.")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="fold 단위 상세 결과 CSV 저장 경로")
    ap.add_argument("--out-summary", default=None, help="조건별 요약 CSV 저장 경로")
    ap.add_argument("--keep-unparsed", action="store_true")
    args = ap.parse_args()

    if args.metric == "ACC" and args.group == "C":
        print("[안내] 그룹 C의 ACC는 gold='Unknown' 기준 정확도입니다. "
              "목적론적 편향을 직접 보려면 --metric TBR_C를 권장합니다.")
    if args.metric in ("TBR_C", "FBR_C") and args.group == "A":
        raise SystemExit(f"{args.metric}는 그룹 C 전용 지표입니다. --group C로 실행하세요.")

    df = load_results(args.results_path, drop_unparsed=not args.keep_unparsed,
                       secondary_labels_path=args.secondary_labels)

    verb_classes = None
    if args.verb_classes:
        if args.verb_classes == ["all"]:
            verb_classes = list(TELIC_VERB_CLASSES)
        elif args.verb_classes == ["none"]:
            verb_classes = None
        else:
            verb_classes = args.verb_classes
            unknown = set(verb_classes) - set(TELIC_VERB_CLASSES)
            if unknown:
                raise SystemExit(f"알 수 없는 verb_class: {sorted(unknown)}. 사용 가능: {TELIC_VERB_CLASSES}")

    secondary_classes = None
    if args.secondary_classes:
        if args.secondary_classes == ["all"]:
            secondary_classes = list(SECONDARY_VERB_CLASSES)
        else:
            secondary_classes = [normalize_secondary_class(sc) for sc in args.secondary_classes]
            unknown = set(secondary_classes) - set(SECONDARY_VERB_CLASSES)
            if unknown:
                raise SystemExit(f"알 수 없는 secondary_class: {sorted(unknown)}. 사용 가능: {SECONDARY_VERB_CLASSES}")


    if not verb_classes and not secondary_classes:
        raise SystemExit("실행할 verb_classes나 secondary_classes를 1개 이상 지정해야 합니다.")

    fold_level, summary = run_lovo(
        df, group_code=args.group, metric_name=args.metric,
        verb_classes=verb_classes, secondary_classes=secondary_classes,
        models=args.models, strategies=args.strategies,
    )

    pd.set_option("display.width", 160)
    print(f"\n=== LOVO 요약 (group={args.group}, metric={args.metric}) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n해석: fold_range/fold_std가 클수록 그 유형·조건에서 결론이 동사 1개에 좌우됨.")
    print("n_source_verbs가 작은 유형일수록 이 값이 커지기 쉬우니,")
    print("표본이 큰 유형(예: Change_of_State=44)의 fold_range와 나란히 비교해 보세요.")

    if args.out:
        fold_level.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[저장됨] {args.out}")
    if args.out_summary:
        summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_summary}")


if __name__ == "__main__":
    _cli()

