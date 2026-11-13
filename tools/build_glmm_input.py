# -*- coding: utf-8 -*-
"""glmm/*.R 스크립트가 읽는 glmm_input.csv를 결과 JSONL에서 만든다.

사용: python tools/build_glmm_input.py results/main_clean.jsonl 비완결동사_이중라벨링_매핑.csv glmm_input.csv
열: id, g, verb, verb_class, sec, model, strategy, predicted_label, gold_label, correct
  - g   : 그룹 문자(A–D)
  - sec : 2차(한국어) 라벨. 매핑 CSV의 '한국어분류' (완결동사 그룹 A·C에만 해당)
  - correct : predicted_label == gold_label 이면 1
"""
import json
import sys

import pandas as pd

res_path, map_path, out_path = (sys.argv[1:4] + [None, None, None])[:3]
res_path = res_path or "results/main_clean.jsonl"
map_path = map_path or "비완결동사_이중라벨링_매핑.csv"
out_path = out_path or "glmm_input.csv"

rows = [json.loads(l) for l in open(res_path, encoding="utf-8") if l.strip()]
d = pd.DataFrame(rows)
d = d[~d["raw_response"].astype(str).str.startswith("ERROR")]
d["g"] = d["group"].str[0]
mp = pd.read_csv(map_path).drop_duplicates("동사").rename(columns={"동사": "verb", "한국어분류": "sec"})[["verb", "sec"]]
d = d.merge(mp, on="verb", how="left")
d.loc[~d["g"].isin(["A", "C"]), "sec"] = None
d["correct"] = (d["predicted_label"] == d["gold_label"]).astype(int)
cols = ["id", "g", "verb", "verb_class", "sec", "model", "strategy", "predicted_label", "gold_label", "correct"]
d[cols].to_csv(out_path, index=False, encoding="utf-8")
print(f"[saved] {out_path}: {len(d)} rows, models={sorted(d['model'].unique())}")
