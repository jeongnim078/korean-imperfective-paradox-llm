# -*- coding: utf-8 -*-
"""Colab 대리 모델 재현 노트북(proxy_replication_colab.ipynb) 생성 스크립트."""
import json
import pandas as pd
import nbformat as nbf

items = json.load(open("dataset_from_results.json", encoding="utf-8"))
mp = pd.read_csv("mapping.csv").set_index("동사")["한국어분류"].to_dict()
C = [dict(id=r["id"], verb_class=r["verb_class"], secondary_class=mp.get(r["verb"]), verb=r["verb"],
          premise=r["premise"], hypothesis=r["hypothesis"]) for r in items if r["id"].startswith("C_")]
assert len(C) == 100 and all(c["secondary_class"] for c in C)
C_JSON = json.dumps(C, ensure_ascii=False, indent=0)

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 대리 모델 재현 실험 (1단계) — Qwen3-4B-Instruct-2507, 그룹 C

**목적**: 본실험의 Qwen3-30B-A3B-Instruct-2507에서 나타난 "DAP → 모호한 완결 문항(그룹 C)에서 '거짓' 과신" 패턴이
Colab 무료 T4에 올라가는 같은 계열의 소형 모델에서도 재현되는지 확인한다. 재현되면 2단계(logit lens)를 이 모델로 진행한다.

**설계**: 그룹 C 100문항 × {Zero-shot, DAP} = 200회 생성. 프롬프트·파서는 본실험(`prompts.py`, `parser.py`)과 동일. 그리디 디코딩(temperature 0에 해당).

**사전 등록한 판정 기준** (결과를 보기 전에 고정):
- **재현**: FBR_C(DAP) − FBR_C(Zero-shot) ≥ 0.15 **그리고** 같은 문항 쌍에 대한 정확 McNemar 검정 p < 0.05 (거짓 응답 증가 방향)
- **부분 재현**: ACC_C는 유의하게 떨어졌지만 오답이 '거짓'이 아닌 다른 방향인 경우 → 결과는 보고하되 2단계 진행 여부는 다시 판단
- **미재현**: 그 외 → 메커니즘 분석은 향후 과제로 넘기고, "모델 크기에 따라 달라지는 현상"으로 보고

참고(본실험 Qwen3-30B, 그룹 C): Zero-shot ACC 0.80 / TBR 0.15 / FBR 0.05, DAP ACC 0.47 / TBR 0.11 / FBR 0.42

**실행 방법**: 메뉴 → 런타임 → 런타임 유형 변경 → **T4 GPU** 선택 후, 위에서부터 순서대로 실행. 전체 약 10~15분.
세션이 끊겨도 결과가 한 줄씩 저장되므로, 처음부터 다시 실행하면 이미 끝난 문항은 건너뛴다(Google Drive 저장 사용 시).""")

md("## 0. 설정")
code("""# ===== 설정 (필요하면 여기만 수정) =====
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"   # 대안: "Qwen/Qwen3-8B" (+ LOAD_MODE="4bit")
LOAD_MODE = "fp16"                          # "fp16" 또는 "4bit". fp16에서 출력이 깨지면 "4bit"로
MODEL_KEY = "qwen4b_proxy"                  # 결과 파일의 model 필드 값
USE_DRIVE = True                            # True면 Google Drive에 저장(세션 끊겨도 이어서 실행 가능)
MAX_NEW_TOKENS = 16
STRATEGIES = ["zero-shot", "dap"]""")

md("## 1. GPU 확인 및 패키지 설치")
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U "transformers>=4.51.0" accelerate
if LOAD_MODE == "4bit":
    !pip install -q -U bitsandbytes""")

md("## 2. 저장 위치")
code("""import os, json
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_DIR = '/content/drive/MyDrive/imperfective_proxy'
else:
    OUT_DIR = '/content/imperfective_proxy'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, f'{MODEL_KEY}_groupC.jsonl')
print('결과 파일:', OUT_PATH)""")

md("## 3. 데이터 (그룹 C 100문항, 본실험과 동일)")
code("ITEMS_C = json.loads(r'''" + C_JSON + "''')\nprint(len(ITEMS_C), '문항 |', ITEMS_C[0])")

md("## 4. 프롬프트와 파서 (본실험 `prompts.py`, `parser.py`와 동일)")
code('''PROMPTS = {
    "zero-shot": {
        "system": """당신은 엄격한 논리학자입니다. 당신의 임무는 전제(Premise)가 주어졌을 때 가설(Hypothesis)이 반드시 참인지 판단하는 것입니다.
전제만으로 가설이 반드시 참이면 "참"을 출력하십시오.
가설이 전제와 모순되면 "거짓"을 출력하십시오.
가설이 참일 수도 있지만 전제에서 명시적으로 보장되지 않으면 "모름"을 출력하십시오.
상식적 추측을 사용하지 마십시오. 오직 제공된 텍스트만 사용하십시오.""",
        "user_format": """입력: 전제: {premise} 가설: {hypothesis}
다음 중 하나로만 답하십시오: "참", "거짓", 또는 "모름".""",
    },
    "dap": {
        "system": """당신은 엄격한 논리학자입니다. 당신의 임무는 전제(Premise)가 주어졌을 때 가설(Hypothesis)이 반드시 참인지 판단하는 것입니다.
전제만으로 가설이 반드시 참이면 "참"을 출력하십시오.
가설이 전제와 모순되면 "거짓"을 출력하십시오.
가설이 참일 수도 있지만 전제에서 명시적으로 보장되지 않으면 "모름"을 출력하십시오.

중요한 언어학적 규칙: 활동동사(activity)와 완수동사(accomplishment)의 차이에 주의하십시오.
- 완수동사/목표지향적 행위(예: "짓고 있었다", "쓰고 있었다")의 경우, 진행형이 완료를 의미하지 않습니다.
- 활동동사(예: "달리고 있었다", "노래하고 있었다")의 경우, 그 과정 자체가 행동이 일어났음을 의미합니다.
상식적 추측을 사용하지 마십시오. 오직 제공된 텍스트만 사용하십시오.""",
        "user_format": """입력: 전제: {premise} 가설: {hypothesis}
다음 중 하나로만 답하십시오: "참", "거짓", 또는 "모름".""",
    },
}

LABEL_MAP = {"참": "True", "거짓": "False", "모름": "Unknown"}

def parse_simple(raw_text):
    text = (raw_text or "").strip()
    found = [ko for ko in LABEL_MAP if ko in text]
    if len(found) == 1:
        return {"label": LABEL_MAP[found[0]], "parse_ok": True}
    return {"label": None, "parse_ok": False}

assert parse_simple("참")["label"] == "True" and parse_simple('"모름"')["label"] == "Unknown"
assert parse_simple("참 또는 거짓")["parse_ok"] is False''')

md("## 5. 모델 불러오기")
code("""import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if LOAD_MODE == "4bit":
    from transformers import BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=qcfg, device_map="cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print('레이어 수:', model.config.num_hidden_layers, '| GPU 메모리 사용:', round(torch.cuda.memory_allocated()/1e9, 1), 'GB')

# 라벨 단어의 첫 토큰 id (2단계 logit lens에서도 같은 값을 씀)
LABEL_TOKENS = {}
for ko in LABEL_MAP:
    ids = tok.encode(ko, add_special_tokens=False)
    LABEL_TOKENS[ko] = ids[0]
    print(f'{ko}: 토큰 {ids} -> {[tok.decode([i]) for i in ids]}')
assert len(set(LABEL_TOKENS.values())) == 3, '세 라벨의 첫 토큰이 겹칩니다 — 2단계 전에 토큰 처리 방식을 바꿔야 합니다.'""")

md("## 6. 생성 함수 + 동작 확인 (2문항)")
code("""@torch.no_grad()
def run_one(strategy, premise, hypothesis):
    p = PROMPTS[strategy]
    messages = [{"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user_format"].format(premise=premise, hypothesis=hypothesis)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)  # 2507 Instruct에서는 무시됨, Qwen3-8B 대비용
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=tok.eos_token_id)
    gen_ids = out.sequences[0, enc.input_ids.shape[1]:]
    raw = tok.decode(gen_ids, skip_special_tokens=True)
    # 첫 생성 위치에서 세 라벨 토큰의 확률 (전체 어휘 softmax 기준)
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    label_probs = {LABEL_MAP[ko]: round(probs[t].item(), 5) for ko, t in LABEL_TOKENS.items()}
    first_tok = tok.decode(gen_ids[:1])
    return raw, label_probs, first_tok

for it in ITEMS_C[:2]:
    for s in STRATEGIES:
        raw, lp, ft = run_one(s, it["premise"], it["hypothesis"])
        print(s, '|', it["premise"], '->', repr(raw), '| 첫 토큰', repr(ft), '|', lp)
print('\\n위 출력이 "참/거짓/모름" 중 하나로 나오면 정상. 깨진 문자나 빈 출력이면 LOAD_MODE="4bit"로 바꿔 5번부터 다시 실행.')""")

md("## 7. 본 실행 (200회, 한 줄씩 저장 · 이어서 실행 가능)")
code("""import time
done = set()
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line); done.add((r['strategy'], r['id']))
print('이미 완료:', len(done))

t0 = time.time(); n_new = 0
with open(OUT_PATH, 'a', encoding='utf-8') as f:
    for s in STRATEGIES:
        for it in ITEMS_C:
            if (s, it['id']) in done:
                continue
            raw, lp, ft = run_one(s, it['premise'], it['hypothesis'])
            parsed = parse_simple(raw)
            rec = dict(id=it['id'], group='C_Ambiguous_Accomplishment', verb_class=it['verb_class'],
                       secondary_class=it['secondary_class'], verb=it['verb'],
                       premise=it['premise'], hypothesis=it['hypothesis'], gold_label='Unknown',
                       model=MODEL_KEY, model_name=MODEL_ID, strategy=s,
                       predicted_label=parsed['label'], parse_ok=parsed['parse_ok'],
                       raw_response=raw, reasoning=None, possible_interruptions=None,
                       first_token=ft, label_probs=lp, load_mode=LOAD_MODE)
            f.write(json.dumps(rec, ensure_ascii=False) + '\\n'); f.flush()
            n_new += 1
            if n_new % 20 == 0:
                print(f'{n_new}건 완료 ({time.time()-t0:.0f}초)')
print(f'끝. 새로 {n_new}건, {time.time()-t0:.0f}초')""")

md("## 8. 지표 계산 및 사전 등록 기준 판정")
code("""import pandas as pd
from scipy.stats import binomtest

df = pd.read_json(OUT_PATH, lines=True)
df = df[df.strategy.isin(STRATEGIES)]
print('파싱 성공률:', df.groupby('strategy').parse_ok.mean().round(3).to_dict())

def metrics(d):
    ok = d[d.parse_ok]
    return pd.Series({'n': len(d), 'parse_ok': d.parse_ok.mean(),
                      'ACC_C': (ok.predicted_label == 'Unknown').mean(),
                      'TBR_C': (ok.predicted_label == 'True').mean(),
                      'FBR_C': (ok.predicted_label == 'False').mean()})

summary = df.groupby('strategy').apply(metrics).round(3)
ref = pd.DataFrame({'ACC_C': [0.80, 0.47], 'TBR_C': [0.15, 0.11], 'FBR_C': [0.05, 0.42]},
                   index=['zero-shot', 'dap'])
print('\\n[대리 모델]'); print(summary)
print('\\n[참고: 본실험 Qwen3-30B]'); print(ref)

# 문항 쌍 비교: 같은 문항이 Zero-shot -> DAP에서 '거짓'으로 바뀌었는가 (정확 McNemar)
w = df.pivot_table(index='id', columns='strategy', values='predicted_label', aggfunc='first')
w = w.dropna()
zs_false, dap_false = (w['zero-shot'] == 'False'), (w['dap'] == 'False')
b = int((~zs_false & dap_false).sum())   # DAP에서만 거짓
c = int((zs_false & ~dap_false).sum())   # Zero-shot에서만 거짓
p = binomtest(b, b + c, 0.5, alternative='greater').pvalue if (b + c) else 1.0
delta = summary.loc['dap', 'FBR_C'] - summary.loc['zero-shot', 'FBR_C']
print(f'\\nFBR_C 변화(DAP - Zero-shot): {delta:+.3f} | 불일치 쌍 b={b}, c={c} | 정확 McNemar p = {p:.4g}')

c_acc = int(((w['zero-shot'] == 'Unknown') & (w['dap'] != 'Unknown')).sum())
b_acc = int(((w['zero-shot'] != 'Unknown') & (w['dap'] == 'Unknown')).sum())
p_acc = binomtest(c_acc, c_acc + b_acc, 0.5, alternative='greater').pvalue if (b_acc + c_acc) else 1.0

if delta >= 0.15 and p < 0.05:
    verdict = '재현 → 2단계(logit lens)를 이 모델로 진행'
elif p_acc < 0.05:
    verdict = '부분 재현 (ACC는 떨어졌지만 오답 방향이 다름) → 결과 공유 후 2단계 여부 결정'
else:
    verdict = '미재현 → 메커니즘 분석은 향후 과제로, 모델 크기에 따라 달라지는 현상으로 보고'
print('\\n판정:', verdict)""")

md("## 9. 세부 확인 (동사 유형별, 첫 토큰)")
code("""ok = df[df.parse_ok]
print(ok.assign(FBR=ok.predicted_label.eq('False'), TBR=ok.predicted_label.eq('True'))
        .groupby(['strategy', 'secondary_class'])[['FBR', 'TBR']].mean().round(3))
# 첫 생성 토큰이 라벨 토큰인지: 2단계에서 '첫 위치의 라벨 확률'을 쓰려면 대부분 라벨로 시작해야 함
print('\\n첫 토큰 분포:'); print(df.groupby('strategy').first_token.value_counts().head(10))""")

md("## 10. 결과 내려받기\n이 파일(`*_groupC.jsonl`)을 Claude에게 올려 주시면 본실험 결과와 합쳐 분석하고, 2단계 노트북을 이어서 만들어 드립니다.")
code("""if not USE_DRIVE:
    from google.colab import files
    files.download(OUT_PATH)
else:
    print('Google Drive에 저장됨:', OUT_PATH)""")

nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "proxy_replication_colab.ipynb")
print("ok", len(cells), "cells")
