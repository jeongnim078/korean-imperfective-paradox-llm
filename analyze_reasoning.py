# -*- coding: utf-8 -*-
"""
analyze_reasoning.py
====================
[Step 2] 사고 과정(Reasoning Log) 정량 카테고리화 및 오류 유형 분석 스크립트.

모델이 출력한 reasoning(생론 또는 raw_response 내 JSON의 reasoning 필드)을
분석하여 정답(Correct_Logic)과 오답(Error Taxonomy)의 이유를 정량적으로 카테고리화한다.

분류 체계:
-----------
1. Correct_Logic (정답)
   - predicted_label == gold_label 인 정상 응답.

2. Telic_Criteria_Error (완결성/종결점 기준 오적용 오답)
   - 비완결동사(Group D)나 모호 완결사건(Group C)에서 "명시적 종결점이 주어지지 않아 알 수 없다"며 UNKNOWN으로 오판하는 패턴.
   - 키워드: 종결점, 명시되지 않, 완료되었는지 알 수 없, 정해진 종결점, 완료 여부, 결과가 명시 등

3. Interruption_Overinterpretation_Error (중단 사건 과해석 / 미완료 과신 오답)
   - 모호 사건(Group C)이나 비완결동사(Group D)를 보고 "중단되었으므로 실패/거짓이다"라고 FALSE로 오판하는 패턴 (FBR_C의 주요 원인).
   - 키워드: 중단되었으므로, 멈췄으므로, 완료되지 않았, 도중에 중단, 끝내지 못, 미완성 등

4. Boundary_Crossing_Ambiguity (경계통과 상 충돌 오답)
   - '도착하다', '입장하다' 등 전형적 달성동사의 순간적 경계통과 속성과 진행형 어미('-고 있다')의 충돌로 인한 오판.

5. Unknown_Withholding_Error (판단 유보 오답)
   - 정답이 True/False인데 정보 부족이나 모호함을 이유로 UNKNOWN으로 도피한 오답.

6. Other_Error (기타 오답)
   - 상기 오답 패턴에 구체적으로 잡히지 않는 오답.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from data_utils import load_results, filter_df


# ---------------------------------------------------------------------------
# 정규식 패턴 정의
# ---------------------------------------------------------------------------
PATTERNS = {
    "Telic_Criteria_Error": re.compile(
        r"(종결점|명시되지\s*않|종결점이|알\s*수\s*없|완료\s*여부|결과가\s*명시|도달하지\s*못|완료되었는지|정해진\s*종결점|목표가\s*명시)",
        re.IGNORECASE,
    ),
    "Interruption_Overinterpretation_Error": re.compile(
        r"(중단되었으므로|멈췄으므로|완료되지\s*않았|도중에\s*중단|끝내지\s*못|작업을\s*중단|미완성|행동이\s*종결되지|거짓이다|모순된다)",
        re.IGNORECASE,
    ),
    "Boundary_Crossing_Ambiguity": re.compile(
        r"(순간적|경계|도착하는|입장하는|도달하는|순간에|도달했는지)",
        re.IGNORECASE,
    ),
    "Unknown_Withholding_Logic": re.compile(
        r"(모름|알\s*수\s*없|불확실|판단할\s*수\s*없|추단할\s*수\s*없|정보가\s*부족)",
        re.IGNORECASE,
    ),
}


def extract_reasoning_text(row: pd.Series) -> str:
    """df의 row에서 reasoning 텍스트를 최우선 추출."""
    reasoning = row.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    raw = row.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        # JSON 형태인 경우 reasoning 키 파싱 시도
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "reasoning" in parsed and parsed["reasoning"]:
                return str(parsed["reasoning"]).strip()
        except Exception:
            pass
        return raw.strip()

    return ""


def classify_reasoning(row: pd.Series) -> str:
    """단일 행의 reasoning 텍스트 및 정답 여부를 분석하여 카테고리로 분류."""
    text = extract_reasoning_text(row)
    gold = str(row.get("gold_label", ""))
    pred = str(row.get("predicted_label", ""))
    group_code = str(row.get("group_code", ""))
    sec_class = str(row.get("secondary_class", ""))

    # 정답을 맞춘 경우 -> Correct_Logic
    if pred == gold:
        return "Correct_Logic"

    # --- 이하 오답(pred != gold)에 대한 원인 정량 분류 ---
    if not text or text in ("참", "거짓", "모름", "True", "False", "Unknown"):
        # DAP / Zero-shot 처럼 사고 과정(CoT) 문장 없이 단답만 출력된 오답인 경우
        if pred == "False" and group_code in ("C", "D"):
            return "Direct_False_Overconfidence_Error"
        elif pred == "Unknown" and group_code in ("D", "A", "B"):
            return "Direct_Unknown_Withholding_Error"
        elif pred == "True" and group_code == "C":
            return "Direct_True_Hallucination_Error"
        return "Direct_Answer_Other_Error"

    # 1. 비완결동사(D) 또는 모호사건(C)에서 종결점 미명시를 이유로 유보/오판한 경우 -> Telic_Criteria_Error
    if PATTERNS["Telic_Criteria_Error"].search(text):
        if group_code in ("D", "C") or pred == "Unknown":
            return "Telic_Criteria_Error"

    # 2. 전형적 달성동사 관련 경계통과 상 충돌 오답
    if sec_class == "전형적 달성동사" and PATTERNS["Boundary_Crossing_Ambiguity"].search(text):
        return "Boundary_Crossing_Ambiguity"

    # 3. 중단 사건 과해석 (Group C 또는 D에서 진행 중 중단을 사유로 FALSE로 과단한 경우)
    if PATTERNS["Interruption_Overinterpretation_Error"].search(text):
        if pred == "False" and group_code in ("C", "D"):
            return "Interruption_Overinterpretation_Error"

    # 4. 판단 유보 오답 (Group D나 A/B에서 UNKNOWN을 선택한 오답)
    if pred == "Unknown" and PATTERNS["Unknown_Withholding_Logic"].search(text):
        return "Unknown_Withholding_Error"

    return "Other_CoT_Error"



def analyze_reasoning_df(df: pd.DataFrame) -> pd.DataFrame:
    """df 전체에 reasoning_category 컬럼을 추가."""
    df = df.copy()
    df["reasoning_text"] = df.apply(extract_reasoning_text, axis=1)
    df["reasoning_category"] = df.apply(classify_reasoning, axis=1)
    return df


def summarize_reasoning_by_condition(
    df: pd.DataFrame,
    groupby_cols: list[str] = ["model", "strategy", "group_code"],
) -> pd.DataFrame:
    """조건별 reasoning 오류 카테고리 빈도 및 비율(%) 요약표 생성."""
    if "reasoning_category" not in df.columns:
        df = analyze_reasoning_df(df)

    grouped = df.groupby(groupby_cols + ["reasoning_category"]).size().unstack(fill_value=0)
    totals = grouped.sum(axis=1)

    pct_df = grouped.div(totals, axis=0) * 100

    # 컬럼 이름 가독성 정리
    res = grouped.copy()
    for col in res.columns:
        res[f"{col}_cnt"] = grouped[col]
        res[f"{col}_pct"] = pct_df[col]
        res.drop(columns=[col], inplace=True)

    res["total_n"] = totals
    return res.reset_index()


def main():
    ap = argparse.ArgumentParser(description="Step 2: 사고 과정(Reasoning Log) 정량 카테고리화 및 오류 분석")
    ap.add_argument("results_path", help="run_experiment.py가 만든 jsonl 결과 파일 경로")
    ap.add_argument("--secondary-labels", default=None, help="이중 라벨링 매핑 CSV 경로")
    ap.add_argument("--models", nargs="+", default=None, help="지정한 모델만 필터링")
    ap.add_argument("--strategies", nargs="+", default=None, help="지정한 전략만 필터링")
    ap.add_argument("--out-summary", default=None, help="전체 사고과정 분류 요약 CSV 저장 경로")
    ap.add_argument("--out-by-group", default=None, help="그룹별 사고과정 분류 CSV 저장 경로")
    ap.add_argument("--out-classified-jsonl", default=None, help="분류결과가 포함된 JSONL/CSV 저장 경로")

    args = ap.parse_args()

    df = load_results(args.results_path, secondary_labels_path=args.secondary_labels)

    if args.models:
        df = df[df["model"].isin(args.models)]
    if args.strategies:
        df = df[df["strategy"].isin(args.strategies)]

    classified_df = analyze_reasoning_df(df)

    pd.set_option("display.width", 160)
    print("\n=== [Step 2] 사고 과정(Reasoning Log) 전체 분류 요약 ===")
    summary = summarize_reasoning_by_condition(classified_df, groupby_cols=["model", "strategy"])
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}%"))

    print("\n=== [Step 2] 그룹별(A/B/C/D) 사고 과정 분류 및 오류 분포 ===")
    group_summary = summarize_reasoning_by_condition(classified_df, groupby_cols=["model", "strategy", "group_code"])
    print(group_summary.to_string(index=False, float_format=lambda x: f"{x:.1f}%"))

    # Qwen 그룹 D 세부 분류 출력
    qwen_d = classified_df[(classified_df["model"] == "qwen") & (classified_df["group_code"] == "D")]
    if not qwen_d.empty:
        print("\n=== [특집] Qwen 모델의 Group D (비완결동사) 사고 과정 세부 분류 ===")
        qwen_d_sum = summarize_reasoning_by_condition(qwen_d, groupby_cols=["strategy"])
        print(qwen_d_sum.to_string(index=False, float_format=lambda x: f"{x:.1f}%"))

    # Qwen 그룹 C 세부 분류 출력
    qwen_c = classified_df[(classified_df["model"] == "qwen") & (classified_df["group_code"] == "C")]
    if not qwen_c.empty:
        print("\n=== [특집] Qwen 모델의 Group C (모호완결사건) 사고 과정 세부 분류 ===")
        qwen_c_sum = summarize_reasoning_by_condition(qwen_c, groupby_cols=["strategy"])
        print(qwen_c_sum.to_string(index=False, float_format=lambda x: f"{x:.1f}%"))

    if args.out_summary:
        summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")
        print(f"\n[저장됨] {args.out_summary}")
    if args.out_by_group:
        group_summary.to_csv(args.out_by_group, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_by_group}")
    if args.out_classified_jsonl:
        classified_df.to_csv(args.out_classified_jsonl, index=False, encoding="utf-8-sig")
        print(f"[저장됨] {args.out_classified_jsonl}")


if __name__ == "__main__":
    main()
