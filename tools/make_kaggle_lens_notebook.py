# -*- coding: utf-8 -*-
"""Kaggle용 logit lens 노트북(logit_lens_kaggle.ipynb) 생성."""
import json
import pandas as pd
import nbformat as nbf

items = json.load(open("dataset_from_results.json", encoding="utf-8"))
mp = pd.read_csv("mapping.csv").set_index("동사")["한국어분류"].to_dict()
C = [dict(id=r["id"], verb_class=r["verb_class"], secondary_class=mp.get(r["verb"]), verb=r["verb"],
          premise=r["premise"], hypothesis=r["hypothesis"]) for r in items if r["id"].startswith("C_")]
# 본실험(Ollama, q4_K_M) Qwen3-30B 응답: 30B가 DAP에서 '거짓'으로 바뀐 문항을 표시하는 데 사용
rows = [json.loads(l) for l in open("main.jsonl", encoding="utf-8") if l.strip()]
ref = {(r["id"], r["strategy"]): r["predicted_label"] for r in rows
       if r["model"] == "qwen" and r["id"].startswith("C_") and r["strategy"] in ("zero-shot", "dap")}
for c in C:
    c["ollama30b_zs"] = ref.get((c["id"], "zero-shot"))
    c["ollama30b_dap"] = ref.get((c["id"], "dap"))
assert len(C) == 100 and all(c["secondary_class"] for c in C)
C_JSON = json.dumps(C, ensure_ascii=False)

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Logit lens: Qwen3-30B vs Qwen3-4B — DAP에서 대답이 어디서 갈라지는가

**질문**: 같은 그룹 C 문항, 같은 DAP 프롬프트에서 Qwen3-30B는 '거짓'(미완료 과신), Qwen3-4B는 '모름'(정답)을 고른다.
두 모델의 **층별로 참/거짓/모름 확률이 어떻게 변하는지**를 비교해, 30B의 답이 어느 깊이에서 '거짓' 쪽으로 기우는지 확인한다.

**단계** (위에서부터 순서대로 실행)
- **0단계 — 환경 확인**: GPU 2개, 디스크, 메모리
- **1단계 — 4B**: 행동 재확인(그룹 C × Zero-shot/DAP, 200회) + logit lens
- **2단계 — 30B 재현 확인**: Kaggle 환경(bnb 4비트)에서도 DAP '거짓' 급증이 재현되는지 사전 기준으로 판정
  - 기준(대리 모델 때와 동일, 결과를 보기 전에 고정): FBR_C(DAP) − FBR_C(Zero-shot) ≥ 0.15 **그리고** 정확 McNemar p < 0.05
- **3단계 — 30B logit lens** (2단계에서 재현됐을 때만 해석의 중심으로 사용)
- **4단계 — 비교 그림과 결과 저장**

**Kaggle 설정 (실행 전에 꼭)**
1. 오른쪽 패널 **Session options → Accelerator → GPU T4 x2**
2. **Internet → On** (휴대폰 인증된 계정만 켜짐)
3. 전체 실행 예상: 4B 약 15분 + 30B 다운로드·로드 20~30분 + 30B 실행 약 30분. 결과는 `/kaggle/working/lens_out/`에 저장되고, 세션이 끊겨도 끝난 부분은 건너뛴다.

**본실험과의 차이 (논문 방법론에 적을 것)**: 본실험의 30B는 Ollama q4_K_M(GGUF), 여기서는 transformers + bitsandbytes NF4. 그래서 2단계에서 행동이 같은지 먼저 확인한다.""")

md("## 0. 설정")
code("""RUN_4B = True
RUN_30B = True
MODEL_4B = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_30B_LOCAL = None      # Kaggle 'Add Input'으로 30B 가중치를 붙였다면 그 경로 (예: '/kaggle/input/...'), 아니면 None
OUT_DIR = "/kaggle/working/lens_out"
STRATEGIES = ["zero-shot", "dap"]
MAX_NEW_TOKENS = 16""")

md("""## 0-1. 패키지 (transformers 버전 고정이 중요)
5.x에서는 MoE 전문가 층 구조가 바뀌어 bitsandbytes 4비트 변환이 전문가 층에 적용되지 않을 수 있어, 구조가 검증된 4.57.1로 고정한다.

Kaggle에 미리 깔린 패키지 중 **torchvision·torchaudio·kernels**는 transformers 5.x에 맞춰져 있어 4.57.1과 충돌할 수 있다(모델 코드 불러오기가 실패하면 transformers는 진짜 원인을 숨기고 'Could not find Qwen3ForCausalLM'만 보여준다). 텍스트 모델 실험에는 필요 없으므로 먼저 제거한다.

⚠️ **이 셀을 처음 실행한 뒤에는 반드시 메뉴 Run → Restart & clear cell outputs를 누르고, 이 셀부터 다시 실행**한다. (이미 받은 모델 파일은 지워지지 않는다.)""")
code("""!pip uninstall -y -q torchvision torchaudio kernels
!pip install -q "transformers==4.57.1" "accelerate>=1.0,<2" "bitsandbytes>=0.45" "huggingface_hub>=0.34,<1.0"
import importlib, traceback, transformers, torch
print('transformers', transformers.__version__, '| torch', torch.__version__)
assert transformers.__version__ == '4.57.1', '버전이 다릅니다 → Run → Restart & clear cell outputs 후 이 셀부터 다시 실행'
# 모델 코드를 직접 불러와서, 실패하면 숨겨진 진짜 에러를 그대로 보여준다
for mod in ['transformers.models.qwen3.modeling_qwen3', 'transformers.models.qwen3_moe.modeling_qwen3_moe']:
    try:
        importlib.import_module(mod); print('OK', mod)
    except Exception:
        traceback.print_exc()
        raise SystemExit('위 에러 전체를 복사해서 Claude에게 보내주세요.')""")

md("## 0-2. 환경 확인")
code("""import os, shutil, json, gc, time
import psutil
n_gpu = torch.cuda.device_count()
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i); print(f'GPU{i}: {p.name}, {p.total_memory/1e9:.1f} GB')
print(f'RAM: {psutil.virtual_memory().total/1e9:.1f} GB')
cands = ['/kaggle/tmp', '/tmp', '/root', '/kaggle/working']
free = {}
for d in cands:
    try:
        os.makedirs(d, exist_ok=True); free[d] = shutil.disk_usage(d).free/1e9
    except Exception as e:
        pass
for d, f in free.items(): print(f'디스크 여유 {d}: {f:.0f} GB')
CACHE_DIR = max(free, key=free.get)
print('모델 저장 위치 →', CACHE_DIR)
if RUN_30B and MODEL_30B_LOCAL is None and free[CACHE_DIR] < 66:
    print('\\n⚠️ 30B 원본(약 61GB)을 받을 디스크가 부족합니다. 4B는 그대로 진행되고, 30B는 건너뜁니다.')
    print('   대안: Kaggle "Add Input"에서 Qwen3-30B-A3B-Instruct-2507 가중치를 붙이고 MODEL_30B_LOCAL에 경로 지정')
    RUN_30B = False
assert n_gpu >= 1, 'GPU가 없습니다. Session options에서 GPU T4 x2를 선택하세요.'
if RUN_30B: assert n_gpu >= 2, '30B는 GPU 2개가 필요합니다 (GPU T4 x2).'
os.makedirs(OUT_DIR, exist_ok=True)""")

md("## 0-3. 데이터 (그룹 C 100문항) · 프롬프트 · 파서 — 본실험과 동일")
code("ITEMS_C = json.loads(r'''" + C_JSON + "''')\nprint(len(ITEMS_C), '문항')")
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
    return {"label": LABEL_MAP[found[0]], "parse_ok": True} if len(found) == 1 else {"label": None, "parse_ok": False}''')

md("## 0-4. Logit lens 함수\n각 층의 출력(잔차 흐름)에 모델의 마지막 정규화층과 출력층(lm_head)을 그대로 씌워, **'이 층에서 바로 답한다면' 다음 토큰 확률**을 계산한다. 위치는 프롬프트 마지막(첫 답 토큰을 예측하는 자리). 마지막 층 값이 실제 출력 확률과 같은지 자동으로 검산한다.")
code('''def build_prompt(tok, strategy, it):
    p = PROMPTS[strategy]
    msgs = [{"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user_format"].format(premise=it["premise"], hypothesis=it["hypothesis"])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def label_token_ids(tok):
    ids = {}
    for ko, en in LABEL_MAP.items():
        t = tok.encode(ko, add_special_tokens=False)
        ids[en] = t[0]
        print(f'{ko}: 첫 토큰 {t[0]} {tok.decode([t[0]])!r}')
    assert len(set(ids.values())) == 3, '세 라벨의 첫 토큰이 겹칩니다'
    return ids

@torch.no_grad()
def lens_and_generate(model, tok, strategy, it, lab_ids):
    enc = tok(build_prompt(tok, strategy, it), return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                      # (임베딩, 1층, ..., L층[최종 정규화 적용됨])
    head = model.get_output_embeddings(); norm = model.model.norm
    dev = head.weight.device
    ids = torch.tensor([lab_ids["True"], lab_ids["False"], lab_ids["Unknown"]], device=dev)
    layers = []
    L = len(hs) - 1
    for i in range(1, L + 1):
        h = hs[i][0, -1].to(dev)
        h = h if i == L else norm(h.unsqueeze(0))[0]
        logits = head(h.unsqueeze(0).to(head.weight.dtype))[0].float()
        pr = torch.softmax(logits, -1)
        top = int(pr.argmax())
        lp = pr[ids].tolist()
        layers.append({"layer": i, "p_true": lp[0], "p_false": lp[1], "p_unknown": lp[2], "top_token": tok.decode([top])})
    final = torch.softmax(out.logits[0, -1].float(), -1)[ids.to(out.logits.device)].tolist()
    assert abs(final[1] - layers[-1]["p_false"]) < 1e-3 and abs(final[2] - layers[-1]["p_unknown"]) < 1e-3, '마지막 층 검산 실패'
    gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return raw, layers, L''')

code('''def run_model(tag, model, tok):
    path = os.path.join(OUT_DIR, f"{tag}_groupC_lens.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); done.add((r["strategy"], r["id"]))
    lab_ids = label_token_ids(tok)
    t0 = time.time(); n = 0
    with open(path, "a", encoding="utf-8") as f:
        for s in STRATEGIES:
            for it in ITEMS_C:
                if (s, it["id"]) in done: continue
                raw, layers, L = lens_and_generate(model, tok, s, it, lab_ids)
                pr = parse_simple(raw)
                f.write(json.dumps(dict(model=tag, strategy=s, n_layers=L, raw_response=raw,
                        predicted_label=pr["label"], parse_ok=pr["parse_ok"], layers=layers, **it),
                        ensure_ascii=False) + "\\n"); f.flush()
                n += 1
                if n % 20 == 0: print(f"  {tag}: {n}건 ({time.time()-t0:.0f}초)")
    print(f"{tag} 완료: 새로 {n}건, {time.time()-t0:.0f}초 → {path}")
    return path''')

md("## 1단계. Qwen3-4B (GPU 1개, fp16)")
code('''from transformers import AutoTokenizer, AutoModelForCausalLM
if RUN_4B:
    tok4 = AutoTokenizer.from_pretrained(MODEL_4B, cache_dir=CACHE_DIR)
    m4 = AutoModelForCausalLM.from_pretrained(MODEL_4B, torch_dtype=torch.float16, device_map={"": 0}, cache_dir=CACHE_DIR).eval()
    print('4B 층 수:', m4.config.num_hidden_layers)
    raw, layers, L = lens_and_generate(m4, tok4, "dap", ITEMS_C[0], label_token_ids(tok4))
    print('시험:', repr(raw), '| 마지막 3개 층 P(거짓/모름):', [(round(x["p_false"],3), round(x["p_unknown"],3)) for x in layers[-3:]])
    PATH_4B = run_model("qwen4b", m4, tok4)
    del m4; gc.collect(); torch.cuda.empty_cache()''')

md("## 2단계. Qwen3-30B 불러오기 (bnb NF4, GPU 2개에 나눠 올림)\n원본 가중치(약 61GB)를 받은 뒤 불러오면서 4비트로 바꾼다. 다운로드 10~20분, 로드 10분 안팎. **전문가 층까지 4비트로 바뀌었는지 자동 확인**한다.")
code('''if RUN_30B:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    src = MODEL_30B_LOCAL or MODEL_30B
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok30 = AutoTokenizer.from_pretrained(src, cache_dir=CACHE_DIR)
    m30 = AutoModelForCausalLM.from_pretrained(src, quantization_config=qcfg, device_map="auto",
                                               max_memory={0: "14GiB", 1: "14GiB", "cpu": "24GiB"},
                                               torch_dtype=torch.float16, cache_dir=CACHE_DIR, low_cpu_mem_usage=True).eval()
    n4 = sum(1 for m in m30.modules() if isinstance(m, bnb.nn.Linear4bit))
    nlin = sum(1 for m in m30.modules() if isinstance(m, torch.nn.Linear) and not isinstance(m, bnb.nn.Linear4bit))
    print(f'4비트 층 {n4}개, 4비트가 아닌 Linear {nlin}개 (lm_head·router 등 소수여야 정상)')
    print('장치 배치:', {str(v) for v in m30.hf_device_map.values()})
    for i in range(torch.cuda.device_count()): print(f'GPU{i} 사용 {torch.cuda.memory_allocated(i)/1e9:.1f} GB')
    assert 'cpu' not in {str(v) for v in m30.hf_device_map.values()}, 'CPU로 넘어간 층이 있어 매우 느립니다 — 결과를 알려주세요'
    print('30B 층 수:', m30.config.num_hidden_layers)''')

md("## 3단계. Qwen3-30B 실행 (행동 + logit lens)")
code('''if RUN_30B:
    raw, layers, L = lens_and_generate(m30, tok30, "dap", ITEMS_C[0], label_token_ids(tok30))
    print('시험:', repr(raw))
    PATH_30B = run_model("qwen30b_nf4", m30, tok30)''')

md("## 4단계. 사전 기준 판정 · 본실험(Ollama)과의 일치도")
code('''import pandas as pd, numpy as np
from scipy.stats import binomtest
def load(path):
    return pd.read_json(path, lines=True)
def judge(df, tag):
    ok = df[df.parse_ok]
    m = ok.groupby("strategy").predicted_label.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=["Unknown","True","False"], fill_value=0)
    m.columns = ["ACC_C", "TBR_C", "FBR_C"]
    w = df.pivot(index="id", columns="strategy", values="predicted_label")
    b = int(((w["zero-shot"] != "False") & (w["dap"] == "False")).sum()); c = int(((w["zero-shot"] == "False") & (w["dap"] != "False")).sum())
    p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
    d = m.loc["dap", "FBR_C"] - m.loc["zero-shot", "FBR_C"]
    print(f"\\n[{tag}] 파싱 성공 {df.parse_ok.mean():.0%}"); print(m.round(3))
    print(f"FBR_C 변화 {d:+.2f} | McNemar b={b}, c={c}, p={p:.3g} → {'재현' if d >= 0.15 and p < 0.05 else '미재현'}")
    return w
res = {}
if RUN_4B: res["4b"] = load(os.path.join(OUT_DIR, "qwen4b_groupC_lens.jsonl")); judge(res["4b"], "Qwen3-4B")
if RUN_30B:
    res["30b"] = load(os.path.join(OUT_DIR, "qwen30b_nf4_groupC_lens.jsonl")); w = judge(res["30b"], "Qwen3-30B NF4")
    ref = pd.DataFrame(ITEMS_C).set_index("id")
    for s, col in [("zero-shot", "ollama30b_zs"), ("dap", "ollama30b_dap")]:
        print(f"본실험(Ollama q4_K_M)과 문항별 일치 — {s}: {(w[s] == ref[col]).mean():.0%}")''')

md("## 5단계. 비교 그림: 층 깊이별 P(거짓)과 P(모름)\n두 모델의 층 수(4B 36층, 30B 48층)가 달라 가로축은 **상대 깊이(0~1)**로 맞춘다. 실선 = DAP, 점선 = Zero-shot. 세 라벨 확률을 합이 1이 되도록 다시 나눈 값(라벨 간 상대 확률)을 쓴다.")
code('''import matplotlib.pyplot as plt
def layer_frame(df):
    rows = []
    for _, r in df.iterrows():
        for x in r["layers"]:
            tot = x["p_true"] + x["p_false"] + x["p_unknown"] + 1e-12
            rows.append(dict(model=r["model"], strategy=r["strategy"], id=r["id"], depth=x["layer"] / r["n_layers"], layer=x["layer"],
                             false=x["p_false"]/tot, unknown=x["p_unknown"]/tot, true=x["p_true"]/tot, mass=tot,
                             final=r["predicted_label"], ollama_dap=r["ollama30b_dap"]))
    return pd.DataFrame(rows)
if res:
    LF = pd.concat([layer_frame(d) for d in res.values()])
    LF.to_csv(os.path.join(OUT_DIR, "layer_frame.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, lab in zip(axes, ["false", "unknown"]):
        for (mdl, s), g in LF.groupby(["model", "strategy"]):
            mean = g.groupby("depth")[lab].mean()
            ax.plot(mean.index, mean.values, ls="-" if s == "dap" else "--", label=f"{mdl} / {s}")
        ax.set_title({"false": "P(False) - non-completion overconfidence", "unknown": "P(Unknown) - correct"}[lab]); ax.set_xlabel("relative depth (0 = input, 1 = output)")
    axes[0].set_ylabel("share among 3 labels (mean of 100 Group C items)"); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "lens_compare.png"), dpi=150); plt.show()
    print(LF.groupby(["model","strategy"]).apply(lambda g: g[g.depth > 0.5].mass.mean()).round(3).rename("상위 절반 층의 라벨 토큰 확률 합"))''')

md("## 6단계. 결과 내려받기\n오른쪽 **Output** 패널에서 `lens_out` 폴더의 파일들을 내려받거나, 아래 셀로 zip 하나로 묶는다. 이 zip을 Claude에게 올려 주시면 층별 분기점 분석과 논문용 그림을 만든다.")
code('''shutil.make_archive("/kaggle/working/lens_out", "zip", OUT_DIR)
print("/kaggle/working/lens_out.zip", round(os.path.getsize("/kaggle/working/lens_out.zip")/1e6, 1), "MB")''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "logit_lens_kaggle.ipynb")
print("ok", len(cells), "cells")
