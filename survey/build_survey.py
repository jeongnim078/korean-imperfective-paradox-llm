# -*- coding: utf-8 -*-
"""원어민 검수 서베이용 층화 샘플 추출 + 응답 xlsx / 정답키 생성.

사용: python build_survey.py imperfective_paradox_ko_full.json mapping.csv main.jsonl
산출: survey_sample_key.csv (연구자용 정답키), 원어민검수_설문지.xlsx (응답자용)
"""
import json, random, sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 20261013
data_path, map_path, res_path = sys.argv[1:4]
items = pd.DataFrame(json.load(open(data_path, encoding="utf-8")))
items["g"] = items["group"].str[0]
mp = pd.read_csv(map_path).rename(columns={"동사": "verb", "한국어분류": "sec", "상위유형": "primary_raw"})
items = items.merge(mp[["verb", "sec", "primary_raw", "문맥의존여부"]], on="verb", how="left")

# 모델 오답 정보(우선순위 판단·사후 분석용, 응답자에게는 노출하지 않음)
res = pd.DataFrame([json.loads(l) for l in open(res_path, encoding="utf-8") if l.strip()])
res = res[res.model.isin(["gpt", "qwen"]) & res.parse_ok]
res["wrong"] = (res.predicted_label != res.gold_label).astype(int)
err = res.pivot_table(index="id", columns="model", values="wrong", aggfunc="sum").fillna(0)
err.columns = [f"{c}_wrong_of4" for c in err.columns]
items = items.merge(err, left_on="id", right_index=True, how="left")

# 우선 포함 문항: 파일럿 결함 지적, LOVO 영향 동사, 복합 상위유형, 문맥의존 동사, note 달린 문항
PRIORITY = {
    "A": ["굽다", "다림질하다", "도착하다", "건너다"],
    "C": ["굽다", "따다", "건너다", "부풀리다", "광내다", "다운로드하다", "수확하다",
          "형성하다", "배달하다", "들이다", "삼키다", "갖추다"] ,
    "B": ["돕다", "사냥하다"],
    "D": ["돕다", "흥정하다", "자라다", "쓰다듬다"],
}
# 그룹별 목표 문항 수 (1차 라벨 기준 층화). C는 이동동사 8개 전부 포함.
TARGET = {
    "A": {"Change_of_State": 9, "Creation": 8, "Consumption": 4, "Motion_to_Goal": 4},
    "C": {"Change_of_State": 9, "Creation": 9, "Consumption": 4, "Motion_to_Goal": 8},
    "B": {"Activity": 20},
    "D": {"Activity": 20},
}
REASON = {}
rng = random.Random(SEED)
chosen = []
for g, quota in TARGET.items():
    sub = items[items.g == g]
    for vc, n in quota.items():
        pool = sub[sub.verb_class == vc]
        pri = pool[pool.verb.isin(PRIORITY[g])]
        rest = pool[~pool.verb.isin(PRIORITY[g])]
        pick = list(pri.id)[:n]
        for i in pick:
            REASON[i] = "priority"
        need = n - len(pick)
        extra = rng.sample(list(rest.id), min(need, len(rest)))
        for i in extra:
            REASON[i] = "random"
        chosen += pick + extra

sample = items[items.id.isin(chosen)].copy()
sample["selection"] = sample.id.map(REASON)

# 주의력 점검 문항 (정답이 자명한 문항)
checks = pd.DataFrame([
    {"id": "CHK_1", "premise": "그는 편지 한 통을 다 썼다.", "hypothesis": "그는 편지 한 통을 썼다.", "label": "True"},
    {"id": "CHK_2", "premise": "그녀는 회의에 끝내 참석하지 않았다.", "hypothesis": "그녀는 회의에 참석했다.", "label": "False"},
    {"id": "CHK_3", "premise": "그는 아침에 커피를 마셨다.", "hypothesis": "그는 아침에 두 잔을 마셨다.", "label": "Unknown"},
    {"id": "CHK_4", "premise": "아이들은 운동장에서 한 시간 동안 놀았다.", "hypothesis": "아이들은 놀지 않았다.", "label": "False"},
])
checks["g"] = "CHK"; checks["selection"] = "attention_check"
sample = pd.concat([sample, checks], ignore_index=True)

# 제시 순서 무작위화 (같은 동사의 A/C 쌍이 붙지 않도록 재시도)
order = list(sample.index)
for _ in range(1000):
    rng.shuffle(order)
    verbs = [sample.loc[i, "verb"] for i in order]
    if all(not (isinstance(a, str) and a == b) for a, b in zip(verbs, verbs[1:])):
        break
sample = sample.loc[order].reset_index(drop=True)
sample.insert(0, "문항", range(1, len(sample) + 1))

LABEL_KO = {"True": "참", "False": "거짓", "Unknown": "알 수 없음"}
key_cols = ["문항", "id", "g", "verb", "verb_class", "sec", "primary_raw", "문맥의존여부",
            "selection", "premise", "hypothesis", "label", "gpt_wrong_of4", "qwen_wrong_of4"]
key = sample[key_cols].copy()
key["gold_ko"] = key["label"].map(LABEL_KO)
key.to_csv("survey_sample_key.csv", index=False, encoding="utf-8-sig")

# ---------- 응답용 xlsx ----------
wb = Workbook()
ws0 = wb.active; ws0.title = "안내"
guide = [
    "한국어 문장 판단 설문 (약 40~50분)",
    "",
    "각 문항에는 [전제]와 [가설] 두 문장이 있습니다.",
    "1) 판단: [전제]만 읽었을 때 [가설]이 어떤지 고르세요.",
    "    · 참 — 전제가 사실이면 가설도 반드시 사실이다",
    "    · 거짓 — 전제가 사실이면 가설은 사실일 수 없다",
    "    · 알 수 없음 — 가설이 사실일 수도 있지만, 전제만으로는 확정할 수 없다",
    "   ※ '보통 그렇겠지' 같은 상식적 추측은 하지 말고, 문장에 적힌 내용만으로 판단해 주세요.",
    "2) 자연스러움: 두 문장이 한국어로 얼마나 자연스러운지 1~5점으로 표시해 주세요.",
    "    (1 = 매우 어색함, 3 = 보통, 5 = 매우 자연스러움)",
    "3) 의견: 어색한 표현, 판단이 애매했던 이유 등이 있으면 자유롭게 적어 주세요. (선택)",
    "",
    "· 정답이 정해진 시험이 아닙니다. 직관대로 빠르게 답해 주세요.",
    "· 앞 문항으로 돌아가 답을 바꾸지 말아 주세요.",
    "· 다른 응답자와 상의하지 말아 주세요.",
    "",
    "응답자 이름(또는 코드):",
]
for r, t in enumerate(guide, 1):
    ws0.cell(row=r, column=1, value=t)
ws0["A1"].font = Font(bold=True, size=13)
ws0["B17"].fill = PatternFill("solid", fgColor="FFF2CC")
ws0.column_dimensions["A"].width = 90

ws = wb.create_sheet("응답")
headers = ["문항", "전제", "가설", "판단", "자연스러움(1~5)", "의견(선택)"]
ws.append(headers)
thin = Side(style="thin", color="BBBBBB")
for c in range(1, 7):
    cell = ws.cell(row=1, column=c)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="44546A")
    cell.alignment = Alignment(horizontal="center", vertical="center")
for _, r in sample.iterrows():
    ws.append([int(r["문항"]), r["premise"], r["hypothesis"], None, None, None])
n = len(sample) + 1
dv1 = DataValidation(type="list", formula1='"참,거짓,알 수 없음"', allow_blank=True)
dv2 = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
ws.add_data_validation(dv1); ws.add_data_validation(dv2)
dv1.add(f"D2:D{n}"); dv2.add(f"E2:E{n}")
for row in ws.iter_rows(min_row=2, max_row=n):
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
    row[3].fill = PatternFill("solid", fgColor="FFF2CC")
    row[4].fill = PatternFill("solid", fgColor="FFF2CC")
for col, w in zip("ABCDEF", [6, 48, 38, 14, 14, 30]):
    ws.column_dimensions[col].width = w
ws.freeze_panes = "B2"
wb.save("원어민검수_설문지.xlsx")

summary = key.groupby(["g", "verb_class"], dropna=False).size()
print(summary.to_string())
print("total:", len(key), "| priority:", (key.selection == "priority").sum())
