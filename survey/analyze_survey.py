# -*- coding: utf-8 -*-
"""원어민 검수 응답 분석: Fleiss' kappa, 다수결-정답 일치율, 자연스러움, 모델 오답과의 교차.

사용 (둘 중 하나):
  · 구글 폼 응답 시트 (응답자 1명 = 1행): python analyze_survey.py survey_sample_key.csv 설문응답.csv
      (응답 시트에서 [파일 → 다운로드 → CSV] 또는 .xlsx 그대로)
  · 엑셀 설문지 (응답자 1명 = 파일 1개):   python analyze_survey.py survey_sample_key.csv 응답1.xlsx 응답2.xlsx 응답3.xlsx
산출: survey_item_level.csv (문항별), 콘솔 요약
"""
import re
import sys
import numpy as np
import pandas as pd

KO2EN = {"참": "True", "거짓": "False", "알 수 없음": "Unknown"}
CATS = ["True", "False", "Unknown"]


def _read_table(path, header=0):
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path, header=header, encoding="utf-8-sig")
    return pd.read_excel(path, header=header)


def load_form_export(path):
    """구글 폼 응답 시트: '[n] 가설은…' / '[n] 두 문장의 자연스러움' 열을 문항 단위 긴 표로 변환."""
    raw = _read_table(path)
    name_col = next(c for c in raw.columns if "응답자" in str(c))
    out = []
    for i, (_, row) in enumerate(raw.iterrows(), 1):
        rec = {}
        for col, val in row.items():
            m = re.match(r"^\[(\d+)\]\s*(.*)$", str(col))
            if not m:
                continue
            n, kind = int(m.group(1)), m.group(2)
            slot = rec.setdefault(n, {})
            if "자연스러움" in kind:
                slot["n"] = val
            else:
                slot["j"] = val
        r = pd.DataFrame([{"문항": n, f"j{i}": v.get("j"), f"n{i}": v.get("n")} for n, v in rec.items()])
        r[f"j{i}"] = r[f"j{i}"].map(KO2EN)
        print(f"[응답자 {i}] {row[name_col]}")
        out.append(r)
    return out


def load_excel_sheet(path, i):
    """엑셀 설문지: '문항' 머리글이 있는 행을 찾아 그 아래를 읽는다(구·신 양식 모두)."""
    xl = pd.ExcelFile(path)
    sheet = "설문" if "설문" in xl.sheet_names else "응답"
    probe = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = probe.index[probe.iloc[:, 0].astype(str).str.strip() == "문항"][0]
    r = pd.read_excel(path, sheet_name=sheet, header=hdr)
    r = r.rename(columns={"판단": f"j{i}", "자연스러움(1~5)": f"n{i}"})[["문항", f"j{i}", f"n{i}"]]
    r = r[pd.to_numeric(r["문항"], errors="coerce").notna()].astype({"문항": int})
    r[f"j{i}"] = r[f"j{i}"].map(KO2EN)
    return r


key = pd.read_csv(sys.argv[1])
files = sys.argv[2:]
is_form = len(files) == 1 and any("응답자" in str(c) for c in _read_table(files[0]).columns)
resp = load_form_export(files[0]) if is_form else [load_excel_sheet(f, i) for i, f in enumerate(files, 1)]
df = key.copy()
for r in resp:
    df = df.merge(r, on="문항", how="left")
J = [f"j{i}" for i in range(1, len(resp) + 1)]
N = [f"n{i}" for i in range(1, len(resp) + 1)]

# 주의력 점검
chk = df[df.g == "CHK"]
for j in J:
    print(f"[주의력 점검] {j}: {(chk[j] == chk.label).sum()}/{len(chk)}")

main = df[df.g != "CHK"].copy()
missing = main[J].isna().sum()
if missing.any():
    print("[경고] 미응답:", missing.to_dict())


def fleiss_kappa(table):
    """table: n_items x n_categories 응답 수 행렬 (행마다 평가자 수 동일)."""
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e)


def gwet_ac1(table):
    """Gwet's AC1: 한 범주로 응답이 쏠릴 때(그룹 내 정답이 하나뿐인 경우) kappa의 역설을 보완."""
    n = table.sum(axis=1)[0]
    q = table.shape[1]
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    pi_k = (table / n).mean(axis=0)
    P_e = (pi_k * (1 - pi_k)).sum() / (q - 1)
    return (P_i.mean() - P_e) / (1 - P_e)


full = main.dropna(subset=J)
counts = np.array([[sum(row[j] == c for j in J) for c in CATS] for _, row in full.iterrows()])
print(f"\n전체 (n={len(full)}): Fleiss' kappa = {fleiss_kappa(counts):.3f}, Gwet's AC1 = {gwet_ac1(counts):.3f}")
print("  ※ 그룹 안에서는 정답이 하나뿐이라 kappa가 0 근처로 떨어지는 역설이 생기므로, 그룹별로는 AC1과 쌍별 일치율을 봅니다.")
for g in ["A", "B", "C", "D"]:
    m = (full.g == g).values
    if m.sum() > 1:
        c = counts[m]
        pairwise = (((c ** 2).sum(axis=1) - c.sum(axis=1)) / (c.sum(axis=1) * (c.sum(axis=1) - 1))).mean()
        print(f"  그룹 {g} (n={m.sum()}): AC1 = {gwet_ac1(c):.3f}, 쌍별 일치율 = {pairwise:.1%}")


def majority(row):
    vals = pd.Series([row[j] for j in J]).value_counts()
    return vals.index[0] if vals.iloc[0] >= 2 else "NoMajority"


main["majority"] = main.apply(majority, axis=1)
main["n_agree_gold"] = sum((main[j] == main.label).astype(int) for j in J)
main["majority_matches_gold"] = main.majority == main.label
main["unanimous"] = main[J].nunique(axis=1) == 1
main["naturalness_mean"] = main[N].mean(axis=1)

print(f"\n다수결 = 정답: {main.majority_matches_gold.mean():.1%}  | 만장일치: {main.unanimous.mean():.1%}")
print(main.groupby("g").agg(다수결정답일치=("majority_matches_gold", "mean"),
                            자연스러움평균=("naturalness_mean", "mean"),
                            n=("id", "size")).round(3).to_string())

flag = main[(main.n_agree_gold <= 1) | (main.naturalness_mean < 3)]
print(f"\n검토 필요 문항 (정답 동의 ≤1명 또는 자연스러움 < 3): {len(flag)}개")
print(flag[["문항", "id", "verb", "label", *J, "naturalness_mean"]].to_string(index=False))

# 모델 오답과 사람 불일치의 교차: 사람이 갈린 문항에서 모델도 더 틀렸는가
main["human_contested"] = ~main.unanimous
print("\n사람 판단이 갈린 문항 vs 만장일치 문항의 모델 평균 오답(4전략 중):")
print(main.groupby("human_contested")[["gpt_wrong_of4", "qwen_wrong_of4"]].mean().round(2).to_string())

main.to_csv("survey_item_level.csv", index=False, encoding="utf-8-sig")
print("\n[저장] survey_item_level.csv")
