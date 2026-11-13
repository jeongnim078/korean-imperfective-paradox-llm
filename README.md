# Verifying Teleological Bias in Korean LLMs

Code, data, and results for the IWAIT 2027 abstract
**"Verifying Teleological Bias in Korean LLMs: Lexical Aspect Paradoxes and Error Patterns"**
(Chaemin Jeong, Seunghyeok Hong — SensCore-Lab, Hankuk University of Foreign Studies).

이 저장소는 한국어 불완료상 역설(Imperfective Paradox) 데이터셋으로 LLM의 목적론적 편향을 검증한 실험의 코드·데이터·결과를 담고 있습니다.

## Overview

- **Dataset:** 400 Korean NLI items (`imperfective_paradox_ko_full.json`), 100 per group.

  | Group | Context | Verb type | Gold |
  |---|---|---|---|
  | A | interrupted | accomplishment | False |
  | B | interrupted | activity | True |
  | C | ambiguous | accomplishment | Unknown |
  | D | ambiguous | activity | True |

  Example (C_094 / A_094): 그녀는 강을 건너고 있었다 → 그녀는 강을 건넜다 (**Unknown**); adding "다리가 갑자기 무너져 중간에서 멈춰 섰다" makes it **False**.
- **Verb labels:** primary class following Ma & Miyao (Change_of_State 44, Creation 39, Consumption 9, Motion_to_Goal 8) and a secondary Korean class from six diagnostic tests (`비완결동사_이중라벨링_매핑.csv`).
- **Models:** GPT-5.6-terra (temperature 1, fixed), Gemini-3.1-pro-preview (temperature 0, thinking low), Qwen3-30B-A3B-Instruct-2507 (Ollama q4_K_M, temperature 0). Qwen3-30B NF4 and Qwen3-4B were also run on Group C for replication and logit lens.
- **Prompts** (`prompts.py`): Zero-shot, DAP (definition-aware prompting), CoT, Counterfactual.
- **Metrics:** ACC per group; TBR_C / FBR_C = share of "True" / "False" answers in Group C.

## Repository layout

| Path | Contents |
|---|---|
| `config.py`, `prompts.py`, `clients.py`, `parser.py`, `run_experiment.py` | Experiment pipeline (API calls, parsing, resumable runs) |
| `run_gemini_batch.py` | Optional Gemini Batch API runner (not used for the reported results) |
| `dedupe_results.py` | Keeps the last successful row per (id, model, strategy) |
| `data_utils.py`, `metrics.py`, `bootstrap_ci.py`, `leave_one_verb_out.py`, `analyze_reasoning.py` | Metrics, verb-level cluster bootstrap, LOVO, keyword-based reasoning classification |
| `run_glmm.py` | Earlier GLMM script without random effects — **superseded** by `glmm/refit_glmm.R` |
| `glmm/` | lme4 GLMMs with a random verb intercept (`refit_glmm.R`) and robustness checks (random slope, nAGQ) |
| `notebooks/` | Kaggle logit-lens notebook (Qwen3-30B NF4, Qwen3-4B) and Colab Qwen3-4B replication notebook |
| `survey/` | Native-speaker validation: sample construction, Google Form script, analysis (Fleiss' κ, Gwet's AC1) and anonymized item-level judgments |
| `tools/` | Notebook generators and `build_glmm_input.py` |
| `results/main_clean.jsonl` | All model responses (3 models × 4 strategies × 400 items = 4,800 rows, deduplicated) |
| `results/three_model/` | Final summary tables and bootstrap CIs used in the abstract |
| `results/lens_out/` | Per-layer logit-lens outputs for Group C |
| `results/error_analysis/` | Error-direction analysis and Group A reasoning-type coding |
| `results/lovo/`, `results/reasoning/` | LOVO folds (GPT/Qwen) and reasoning classification |
| `docs/` | Original Korean usage notes for the pipeline and analysis scripts |

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in OPENAI_API_KEY / GEMINI_API_KEY (never commit .env)

python test_parser.py                       # parser check, no API calls
python run_experiment.py --output results/main.jsonl
python dedupe_results.py --input results/main.jsonl --output results/main_clean.jsonl

python metrics.py results/main_clean.jsonl --out-summary results/three_model/summary.csv
python bootstrap_ci.py results/main_clean.jsonl --n-boot 10000 --out results/three_model/bootstrap_3model.csv
python analyze_reasoning.py results/main_clean.jsonl --out-by-group results/reasoning/reasoning_by_group.csv

python tools/build_glmm_input.py results/main_clean.jsonl 비완결동사_이중라벨링_매핑.csv glmm_input.csv
Rscript glmm/refit_glmm.R glmm_input.csv

python survey/analyze_survey.py survey/survey_sample_key.csv <form_responses.csv>
```

Qwen3-30B was served locally with Ollama (`QWEN_BASE_URL`). The logit-lens notebook runs on Kaggle T4×2 with 4-bit NF4 quantization (routers kept unquantized; `transformers==4.57.1`).

## Key results (Group C unless noted)

| | Zero-shot TBR_C | DAP FBR_C | Zero-shot → DAP "False" (up:down) |
|---|---|---|---|
| GPT-5.6-terra | 0.01 | 0.00 | 0:0 |
| Gemini-3.1-pro-preview | 0.00 | 0.00 | 0:0 |
| Qwen3-30B | 0.15 | 0.42 [0.33, 0.51] | 37:0 (exact McNemar p = 1.5×10⁻¹¹) |

- No model answered "True" to any Group A item (0 of 1,200 responses).
- Counterfactual prompting newly failed Group D items in all three models (GPT 8, Gemini 13, Qwen3-30B 90; all "Unknown").
- Native-speaker agreement on 95 sampled items: Fleiss' κ = 0.675 [0.607, 0.736].

## Notes

- `survey/survey_item_level.csv` contains anonymized judgments (`j1`–`j5`); raw form responses with rater names are not included.
- Reasoning categories in `results/reasoning/` come from keyword matching (`analyze_reasoning.py`), not manual coding. `results/error_analysis/A_error_reasoning_types.csv` is the manual coding of Group A CoT errors.
- Decoding settings differ across models (GPT temperature is fixed at 1 by the API), and Qwen3-30B was evaluated only with 4-bit quantization.
