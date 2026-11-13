# -*- coding: utf-8 -*-
"""
bootstrap_ci.py
================
ACC / TBR_C / FBR_C / Delta_AA에 대한 부트스트랩 신뢰구간(percentile CI).

왜 "동사 단위" 클러스터 부트스트랩인가
--------------------------------------
같은 동사(verb)에서 나온 문장들은 서로 독립이 아니다(동사 자체의 특성이
공유됨). 특히 이동동사처럼 표본이 작은 유형(n=8)에서는 문항 단위로
단순 리샘플링하면 실제보다 신뢰구간이 좁게 나와 "안정적인 결과"라는
착시를 줄 수 있다. 이는 leave-one-verb-out과 동일한 문제의식이며,
LOVO가 "동사 하나를 뺐을 때 결과가 얼마나 흔들리는가"를 결정론적으로
보여준다면, 부트스트랩은 "동사 구성이 달라졌다면 있을 법한 값의 분포"를
확률적으로 보여준다는 점에서 상호보완적이다.

그래서 이 스크립트는 문항이 아니라 **동사**를 리샘플링 단위로 삼는다:
    1. 조건(group/verb_class/secondary_class/model/strategy)에 해당하는
       고유 동사 집합 V.
    2. |V|개를 복원추출로 다시 뽑아 동사 목록 V*를 만든다(동사가 여러 번
       뽑히면 그 동사의 모든 문항이 그만큼 중복 포함됨).
    3. V*에 해당하는 문항들로 지표(ACC 또는 TBR/FBR)를 계산 -> 부트스트랩 표본 1개.
    4. 2-3을 n_boot번 반복해 분포를 만들고, percentile CI를 구한다.

Delta_AA는 그룹 D(활동동사)와 그룹 C(완결동사, 특정 verb_class/secondary_class)
라는 서로 다른 동사 풀에서 나오므로, 두 풀을 각각 독립적으로 리샘플링해
같은 반복 회차 안에서 차이를 계산하는 방식으로 결합 CI를 만든다
(두 표본이 독립이라는 가정 하에 유효한 표준적인 방법).

이중 라벨링(2차 라벨) 지원
----------------------------
완결동사 세부 유형은 1차 라벨(verb_class, 원 논문 4분류)과 2차 라벨
(secondary_class, 한국어 문법 테스트 기준 어휘상 범주)이 함께 있을 수 있다.
run_all()에 secondary_classes를 주면(또는 df에 secondary_class 컬럼이
채워져 있으면) TBR_C/FBR_C/Delta_AA를 2차 라벨 기준으로도 동일하게
부트스트랩한다. 1차/2차 축은 서로 완전히 별도의 groupby이며, 각 축의
결과는 출력 표의 verb_class / secondary_class 컬럼으로 구분된다.

사용 예:
    python bootstrap_ci.py results/main.jsonl --n-boot 10000 \\
        --out results/bootstrap_summary.csv

    # 2차 라벨(문법범주) 기준 CI도 함께
    python bootstrap_ci.py results/main.jsonl --n-boot 10000 \\
        --secondary-labels 비완결동사_이중라벨링_매핑.csv \\
        --out results/bootstrap_summary.csv

    # 이동동사만 별도로, 더 촘촘히 보고 싶을 때
    python bootstrap_ci.py results/main.jsonl --verb-classes Motion_to_Goal \\
        --n-boot 20000 --out results/bootstrap_motion.csv
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from data_utils import SECONDARY_VERB_CLASSES, TELIC_VERB_CLASSES, filter_df, load_results, model_strategy_combos, normalize_secondary_class
from metrics import acc_on, fbr_on, tbr_on



def _bootstrap_indices_by_verb(verbs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """고유 동사 배열에서 복원추출로 같은 크기의 동사 리샘플을 만든다."""
    return rng.choice(verbs, size=len(verbs), replace=True)


def bootstrap_metric(
    df: pd.DataFrame,
    metric_fn,
    group_code: str,
    model: str,
    strategy: str,
    verb_class: Optional[str] = None,
    secondary_class: Optional[str] = None,
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """단일 조건에 대해 동사 단위 클러스터 부트스트랩 CI를 계산.

    metric_fn: acc_on, tbr_on, 또는 fbr_on (data:pd.DataFrame -> (value, n))
    verb_class: 1차 라벨 제한. secondary_class: 2차 라벨 제한(둘 다 주면 AND).
    """
    sub = filter_df(df, group_code=group_code, model=model, strategy=strategy,
                     verb_class=verb_class, secondary_class=secondary_class)
    verbs = sub["verb"].dropna().unique()
    point_estimate, n_items = metric_fn(sub)

    if len(verbs) < 2:
        return {
            "group": group_code, "verb_class": verb_class, "secondary_class": secondary_class,
            "model": model, "strategy": strategy,
            "n_verbs": len(verbs), "n_items": n_items,
            "estimate": point_estimate, "boot_mean": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0,
            "note": "동사 수가 2 미만이라 부트스트랩 불가",
        }

    rng = np.random.default_rng(seed)
    # verb -> row indices 매핑을 미리 만들어 반복문에서 재사용 (속도)
    verb_to_rows = {v: sub.index[sub["verb"] == v].to_numpy() for v in verbs}

    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        sampled_verbs = _bootstrap_indices_by_verb(verbs, rng)
        row_idx = np.concatenate([verb_to_rows[v] for v in sampled_verbs])
        boot_df = sub.loc[row_idx]
        val, _ = metric_fn(boot_df)
        boot_vals[b] = val

    alpha = 1 - ci
    lo, hi = np.nanpercentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "group": group_code, "verb_class": verb_class, "secondary_class": secondary_class,
        "model": model, "strategy": strategy,
        "n_verbs": len(verbs), "n_items": n_items,
        "estimate": point_estimate, "boot_mean": float(np.nanmean(boot_vals)),
        "se": float(np.nanstd(boot_vals, ddof=1)),
        "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot,
        "note": "",
    }


def bootstrap_delta_aa(
    df: pd.DataFrame,
    model: str,
    strategy: str,
    verb_class: Optional[str] = None,
    secondary_class: Optional[str] = None,
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Delta_AA = ACC_D - TBR_C 에 대한 결합 부트스트랩 CI.

    그룹 D(활동동사 풀)와 그룹 C(verb_class/secondary_class로 제한된 완결동사 풀)는
    서로 다른 동사 집합이므로 각각 독립적으로 리샘플링한 뒤, 같은 반복
    회차 b에서 ACC_D*_b - TBR_C*_b 를 계산해 분포를 만든다.
    """
    sub_d = filter_df(df, group_code="D", model=model, strategy=strategy)
    sub_c = filter_df(df, group_code="C", model=model, strategy=strategy,
                       verb_class=verb_class, secondary_class=secondary_class)

    verbs_d = sub_d["verb"].dropna().unique()
    verbs_c = sub_c["verb"].dropna().unique()

    point_acc_d, n_d = acc_on(sub_d)
    point_tbr_c, n_c = tbr_on(sub_c)
    point_delta = point_acc_d - point_tbr_c

    if len(verbs_d) < 2 or len(verbs_c) < 2:
        return {
            "model": model, "strategy": strategy,
            "verb_class": verb_class, "secondary_class": secondary_class,
            "n_verbs_D": len(verbs_d), "n_verbs_C": len(verbs_c),
            "estimate": point_delta, "boot_mean": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0,
            "note": "그룹 D 또는 C의 동사 수가 2 미만이라 부트스트랩 불가",
        }

    rng = np.random.default_rng(seed)
    verb_to_rows_d = {v: sub_d.index[sub_d["verb"] == v].to_numpy() for v in verbs_d}
    verb_to_rows_c = {v: sub_c.index[sub_c["verb"] == v].to_numpy() for v in verbs_c}

    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        sv_d = _bootstrap_indices_by_verb(verbs_d, rng)
        sv_c = _bootstrap_indices_by_verb(verbs_c, rng)
        idx_d = np.concatenate([verb_to_rows_d[v] for v in sv_d])
        idx_c = np.concatenate([verb_to_rows_c[v] for v in sv_c])
        acc_d, _ = acc_on(sub_d.loc[idx_d])
        tbr_c, _ = tbr_on(sub_c.loc[idx_c])
        boot_vals[b] = acc_d - tbr_c

    alpha = 1 - ci
    lo, hi = np.nanpercentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "model": model, "strategy": strategy,
        "verb_class": verb_class, "secondary_class": secondary_class,
        "n_verbs_D": len(verbs_d), "n_verbs_C": len(verbs_c),
        "estimate": point_delta, "boot_mean": float(np.nanmean(boot_vals)),
        "se": float(np.nanstd(boot_vals, ddof=1)),
        "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot,
        "note": "",
    }


def run_all(
    df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
    verb_classes: list[str] | None = None,
    secondary_classes: list[str] | None = None,
    models: list[str] | None = None,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """모든 (model,strategy) 조합에 대해 ACC_A/B/C/D, TBR_C, FBR_C, Delta_AA CI를
    1차 라벨(verb_class) 기준으로 계산하고, secondary_class 컬럼이 데이터에
    존재하면(즉 이중 라벨링 매핑을 붙였다면) 2차 라벨 기준으로도 동일하게 계산한다.
    """
    verb_classes = list(TELIC_VERB_CLASSES) if verb_classes is None else verb_classes

    has_secondary = "secondary_class" in df.columns and df["secondary_class"].notna().any()
    if secondary_classes is None:
        secondary_classes = list(SECONDARY_VERB_CLASSES) if has_secondary else []
    elif not has_secondary:
        print("[bootstrap_ci] 안내: secondary_classes가 지정됐지만 df에 secondary_class 값이 "
              "없습니다(load_results에 secondary_labels_path를 주지 않은 것으로 보임). "
              "2차 라벨 기준 CI는 건너뜁니다.")
        secondary_classes = []

    combos = model_strategy_combos(df)
    if models is not None:
        combos = [(m, s) for m, s in combos if m in models]
    if strategies is not None:
        combos = [(m, s) for m, s in combos if s in strategies]

    rows = []
    for model, strategy in combos:
        # 그룹 A, B, D: 세부 유형 구분 없이 전체
        for g, fn in (("A", acc_on), ("B", acc_on), ("D", acc_on)):
            res = bootstrap_metric(df, fn, g, model, strategy, verb_class=None, secondary_class=None,
                                    n_boot=n_boot, seed=seed, ci=ci)
            res["metric"] = "ACC"
            rows.append(res)

        # 그룹 C: 전체 + 1차 라벨(verb_class)별 + 2차 라벨(secondary_class)별, TBR_C/FBR_C
        for metric_name, fn in (("TBR_C", tbr_on), ("FBR_C", fbr_on)):
            res = bootstrap_metric(df, fn, "C", model, strategy, verb_class=None, secondary_class=None,
                                    n_boot=n_boot, seed=seed, ci=ci)
            res["metric"] = metric_name
            rows.append(res)
            for vc in verb_classes:
                res = bootstrap_metric(df, fn, "C", model, strategy, verb_class=vc, secondary_class=None,
                                        n_boot=n_boot, seed=seed, ci=ci)
                res["metric"] = metric_name
                rows.append(res)
            for sc in secondary_classes:
                res = bootstrap_metric(df, fn, "C", model, strategy, verb_class=None, secondary_class=sc,
                                        n_boot=n_boot, seed=seed, ci=ci)
                res["metric"] = metric_name
                rows.append(res)

        # Delta_AA: 전체 + 1차 라벨별 + 2차 라벨별
        res = bootstrap_delta_aa(df, model, strategy, verb_class=None, secondary_class=None,
                                  n_boot=n_boot, seed=seed, ci=ci)
        res["metric"] = "Delta_AA"
        res["group"] = "D-C"
        rows.append(res)
        for vc in verb_classes:
            res = bootstrap_delta_aa(df, model, strategy, verb_class=vc, secondary_class=None,
                                      n_boot=n_boot, seed=seed, ci=ci)
            res["metric"] = "Delta_AA"
            res["group"] = "D-C"
            rows.append(res)
        for sc in secondary_classes:
            res = bootstrap_delta_aa(df, model, strategy, verb_class=None, secondary_class=sc,
                                      n_boot=n_boot, seed=seed, ci=ci)
            res["metric"] = "Delta_AA"
            res["group"] = "D-C"
            rows.append(res)

    out = pd.DataFrame(rows)
    ordered_cols = ["model", "strategy", "metric", "group", "verb_class", "secondary_class",
                     "n_verbs", "n_items", "n_verbs_D", "n_verbs_C",
                     "estimate", "boot_mean", "se", "ci_low", "ci_high", "n_boot", "note"]
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    return out[ordered_cols].sort_values(
        ["model", "strategy", "metric", "verb_class", "secondary_class"], na_position="first"
    ).reset_index(drop=True)


def _cli():
    ap = argparse.ArgumentParser(description="동사 단위 클러스터 부트스트랩 신뢰구간")
    ap.add_argument("results_path")
    ap.add_argument("--secondary-labels", default=None,
                     help="이중 라벨링 매핑 CSV 경로. 주면 2차 라벨(secondary_class) 기준 "
                          "TBR_C/FBR_C/Delta_AA CI도 함께 계산.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--verb-classes", nargs="+", default=None,
                     help="기본은 1차 라벨 4개 세부 유형 전체. 특정 유형만 보려면 이름 나열. "
                          "빈 값을 원하면 --verb-classes (인자 없이)로 생략 불가하니 코드에서 [] 전달 필요.")
    ap.add_argument("--secondary-classes", nargs="+", default=None,
                     help="기본은 2차 라벨 3개(완성동사/정도달성동사/전형적 달성동사) 전체 "
                          "(--secondary-labels를 줬을 때만 적용).")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-unparsed", action="store_true")
    args = ap.parse_args()

    df = load_results(args.results_path, drop_unparsed=not args.keep_unparsed,
                       secondary_labels_path=args.secondary_labels)

    secondary_classes = args.secondary_classes
    if secondary_classes:
        secondary_classes = [normalize_secondary_class(sc) for sc in secondary_classes]

    result = run_all(
        df, n_boot=args.n_boot, seed=args.seed, ci=args.ci,
        verb_classes=args.verb_classes, secondary_classes=secondary_classes,
        models=args.models, strategies=args.strategies,
    )


    pd.set_option("display.width", 160)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    if args.out:
        result.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[저장됨] {args.out}")


if __name__ == "__main__":
    _cli()
