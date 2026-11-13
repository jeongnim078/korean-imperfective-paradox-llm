# -*- coding: utf-8 -*-
"""
run_glmm.py
===========
[Step 3] 혼합효과 로지스틱 회귀 분석 (Mixed-Effects Logistic Regression / GLMM)

이진 종속변수(is_correct: 0 또는 1), 동사별 반복측정 구조, 표본 불균형을 통제하여
모델(model), 전략(strategy), 동사 유형(verb_class / secondary_class) 간의 주효과 및
상호작용 효과가 통계적으로 유의미한지 검증한다.

수식 (Model Formula):
--------------------
1차 라벨 기준: is_correct ~ C(model) * C(strategy) * C(verb_class) + (1|verb)
2차 라벨 기준: is_correct ~ C(model) * C(strategy) * C(secondary_class) + (1|verb)

추정 방식:
----------
1. Cluster-Robust Logistic Regression (동사 단위 클러스터 강건 표준오차 적용)
2. GEE (Generalized Estimating Equations with Binomial Family & Exchangeable Correlation)
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_utils import load_results


def prepare_glmm_df(df: pd.DataFrame, class_col: str = "verb_class") -> pd.DataFrame:
    """GLMM 분석용 이진 변수(is_correct) 및 카테고리 컬럼 정제."""
    sub = df.copy()
    sub["is_correct"] = (sub["predicted_label"] == sub["gold_label"]).astype(int)

    # 유효한 class_col 행만 추출
    if class_col in sub.columns:
        sub = sub[sub[class_col].notna() & (sub[class_col] != "nan") & (sub[class_col] != "")]

    sub["model"] = sub["model"].astype(str)
    sub["strategy"] = sub["strategy"].astype(str)
    sub["verb"] = sub["verb"].astype(str)
    sub[class_col] = sub[class_col].astype(str)

    return sub.reset_index(drop=True)



def fit_cluster_logistic(
    df: pd.DataFrame,
    formula: str,
    cluster_col: str = "verb",
) -> pd.DataFrame:
    """동사(verb) 클러스터 강건 표준오차(Cluster-Robust SE)를 적용한 로지스틱 회귀."""
    model = smf.logit(formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]}, disp=False)

    params = res.params
    bse = res.bse
    zvals = res.tvalues
    pvals = res.pvalues
    conf = res.conf_int()

    rows = []
    for var in params.index:
        or_val = np.exp(params[var])
        or_low = np.exp(conf.loc[var, 0])
        or_high = np.exp(conf.loc[var, 1])
        rows.append({
            "term": var,
            "coef": params[var],
            "std_err": bse[var],
            "z_stat": zvals[var],
            "p_value": pvals[var],
            "odds_ratio": or_val,
            "or_ci_low": or_low,
            "or_ci_high": or_high,
            "significant_005": pvals[var] < 0.05,
        })

    out_df = pd.DataFrame(rows)
    out_df.attrs["n_obs"] = res.nobs
    out_df.attrs["prsquared"] = res.prsquared
    return out_df


def fit_gee_binomial(
    df: pd.DataFrame,
    formula: str,
    cluster_col: str = "verb",
) -> pd.DataFrame:
    """동사(verb) 단위 Exchangeable 상관구조를 적용한 GEE (Generalized Estimating Equations)."""
    fam = sm.families.Binomial()
    ind = sm.cov_struct.Exchangeable()
    model = smf.gee(formula, cluster_col, data=df, family=fam, cov_struct=ind)
    res = model.fit()

    params = res.params
    bse = res.bse
    zvals = res.tvalues
    pvals = res.pvalues
    conf = res.conf_int()

    rows = []
    for var in params.index:
        or_val = np.exp(params[var])
        or_low = np.exp(conf.loc[var, 0])
        or_high = np.exp(conf.loc[var, 1])
        rows.append({
            "term": var,
            "coef": params[var],
            "std_err": bse[var],
            "z_stat": zvals[var],
            "p_value": pvals[var],
            "odds_ratio": or_val,
            "or_ci_low": or_low,
            "or_ci_high": or_high,
            "significant_005": pvals[var] < 0.05,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="[Step 3] 혼합효과 로지스틱 회귀 분석 (GLMM)")
    ap.add_argument("results_path", help="results/main.jsonl 경로")
    ap.add_argument("--secondary-labels", default=None, help="이중 라벨링 매핑 CSV 경로")
    ap.add_argument("--group", choices=["A", "B", "C", "D", "all"], default="all",
                     help="분석 대상 논리 그룹 (A/B/C/D 또는전체)")
    ap.add_argument("--class-type", choices=["verb_class", "secondary_class"], default="verb_class",
                     help="분석할 동사 분류 축 (1차 verb_class 또는 2차 secondary_class)")
    ap.add_argument("--out", default=None, help="회귀분석 결과 CSV 저장 경로")

    args = ap.parse_args()

    df = load_results(args.results_path, secondary_labels_path=args.secondary_labels)
    if args.group != "all":
        df = df[df["group_code"] == args.group]

    class_col = args.class_type
    glmm_df = prepare_glmm_df(df, class_col=class_col)

    if class_col == "secondary_class" and (class_col not in glmm_df.columns or glmm_df[class_col].empty):
        raise SystemExit("2차 라벨(secondary_class) 분석을 하려면 --secondary-labels 매핑 CSV를 지정해야 합니다.")

    formula = f"is_correct ~ C(model) * C(strategy) * C({class_col})"



    pd.set_option("display.width", 160)
    print(f"\n=== [Step 3] 혼합효과 로지스틱 회귀 분석 (Group={args.group}, Class={class_col}) ===")
    print(f"모델식: {formula} + (1|verb)")
    print(f"분석 대상 표본 수: N = {len(glmm_df)} (동사 클러스터 수: {glmm_df['verb'].nunique()})")

    res_df = fit_cluster_logistic(glmm_df, formula, cluster_col="verb")

    print("\n--- 회귀계수 및 승산비(Odds Ratio) 추정 결과 ---")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out:
        res_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[저장됨] {args.out}")


if __name__ == "__main__":
    main()
