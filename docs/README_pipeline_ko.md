# 미완료상 역설 한국어 데이터셋 — 행동 실험 파이프라인

## 파일 구성
- `config.py` — 모델 이름, 경로, temperature 등 전역 설정
- `prompts.py` — 프롬프트 4종(Zero-shot/DAP/CoT/반사실적) 한국어 템플릿
- `clients.py` — OpenAI(gpt-5.6-terra), Gemini(gemini-3.1-pro-preview) 호출 래퍼 (재시도 포함)
- `parser.py` — 모델 응답에서 참/거짓/모름 라벨 추출
- `run_experiment.py` — 메인 실행 스크립트 (재시작 가능, 파일럿 지원)
- `test_parser.py` — API 호출 없이 파서 로직만 검증하는 테스트

## 1. 준비

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

`imperfective_paradox_ko_full.json`(400개 데이터셋)을 이 폴더에 함께 두세요.
(경로가 다르면 `--dataset` 옵션으로 지정 가능)

## 2. 파서만 먼저 검증 (API 호출 없음, 무료)

```bash
python test_parser.py
```

9개 케이스가 모두 OK로 나오면 파싱 로직은 정상입니다.

## 3. 파일럿 테스트 (권장 — 본실행 전에 꼭 먼저)

데이터셋 앞에서 20문항만 뽑아 두 모델 × 4전략으로 최대 160회 호출:

```bash
python run_experiment.py --limit 20 --output results/pilot.jsonl
```

`results/pilot.jsonl`을 열어서:
- `predicted_label`이 `True`/`False`/`Unknown`으로 잘 채워졌는지
- `parse_ok`가 `false`인 행이 있는지 (있으면 해당 `raw_response`를 보고 프롬프트나 파서를 조정)
- CoT/반사실적 전략의 `reasoning`/`possible_interruptions`가 그럴듯하게 나오는지

를 확인하세요.

## 4. 본실행 (400문항 전체)

```bash
python run_experiment.py --output results/main.jsonl
```

- 문항당 호출은 `모델 2개 × 전략 4개 = 8회`, 전체 `400 × 8 = 3,200회` 호출입니다.
- 중간에 인터넷이 끊기거나 rate limit에 걸려도, **같은 명령을 다시 실행하면** 이미
  `results/main.jsonl`에 기록된 (id, model, strategy) 조합은 자동으로 건너뛰고 이어서
  진행합니다. 파일을 지우지 않는 한 안전합니다.
- 특정 모델/전략만 다시 돌리고 싶으면:
  ```bash
  python run_experiment.py --output results/main.jsonl --models gemini --strategies cot
  ```

## 5. 결과 형식 (JSONL, 한 줄에 한 호출 결과)

```json
{
  "id": "A_001",
  "group": "A_Interrupted_Accomplishment",
  "verb_class": "Creation",
  "verb": "짓다",
  "premise": "...",
  "hypothesis": "...",
  "gold_label": "False",
  "model": "gpt",
  "model_name": "gpt-5.6-terra",
  "strategy": "zero-shot",
  "predicted_label": "False",
  "parse_ok": true,
  "raw_response": "거짓",
  "reasoning": null,
  "possible_interruptions": null
}
```

- `gold_label`: 데이터셋에 저장된 정답
- `predicted_label`: 모델 응답에서 파싱한 라벨
- `parse_ok`: 파싱이 확실히 성공했는지 (false면 `raw_response`를 사람이 직접 확인 필요)
- `reasoning`/`possible_interruptions`: CoT/반사실적 전략에서만 채워짐

## 6. Qwen(오픈웨이트) 로컬 구동 준비물

`config.py` / `clients.py`가 Qwen3-30B-A3B-Instruct-2507을 Ollama로 로컬 구동하도록
이미 설정되어 있습니다(README.md 5절 대신 이 섹션을 따르세요). CPU 32GB RAM 기준.

```bash
# 1) https://ollama.com 에서 Ollama 설치 (Windows/Mac/Linux 모두 지원)

# 2) 모델 다운로드 (약 19GB, 한 번만 받으면 됨)
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M

# 3) Ollama 서버가 켜져 있는지 확인 (보통 설치 시 자동 실행됨)
#    안 켜져 있다면 별도 터미널에서:
ollama serve
```

그 다음은 기존과 동일하게 실행하면 됩니다(`.env`에 OPENAI/GEMINI 키가 없어도
Qwen만 따로 테스트 가능):

```bash
# 속도 감 잡기용 초소형 파일럿 (문항 2개, Qwen만)
python run_experiment.py --models qwen --strategies zero-shot --limit 2 --output results/qwen_speedtest.jsonl

# 통상 파일럿 (20문항, 세 모델 x 4전략)
python run_experiment.py --limit 20 --output results/pilot.jsonl

# 본실행
python run_experiment.py --output results/main.jsonl
```

- 첫 호출이 끝나면 콘솔에 `[Qwen 로컬] N초, M 토큰 (~X tok/s)`가 매 호출마다 출력되고,
  실행이 끝나면 `[Qwen 로컬 사용량 요약]`에서 본실행(1,600회) 예상 소요 시간을 추정해 보여줍니다.
  너무 오래 걸릴 것 같으면 `--models qwen`만 따로, 밤사이 돌려두는 것을 권장합니다.
- 메모리 부족(스왑 발생, 체감상 급격히 느려짐)이 생기면 `config.py`의
  `MODELS["qwen"]`을 `"qwen3:30b-a3b-instruct-2507-q4_0"`로 바꿔보세요(용량↓, 품질 소폭↓).
- 이 모델은 gemini-3.1-pro-preview / gpt-5.6-terra와 동급 성능이 아닙니다(로컬 CPU
  구동이 가능한 선에서 고른 절충안). 논문/보고서의 한계(Limitations) 절에 이 점을
  명시하는 것을 권장합니다.

## 7. Qwen(오픈웨이트) 추가 시 — 원본 안내 (참고용, 이미 완료됨)

~~`clients.py`에 `call_qwen(system_prompt, user_prompt)` 함수를 추가하고
`CALL_FUNCS` 딕셔너리에 `"qwen": call_qwen`만 넣으면...~~ → 이미 반영되어 있습니다.
Qwen을 다른 오픈웨이트 모델로 교체하고 싶을 때만 `call_qwen`을 참고해서 고치면 됩니다.

## 8. 다음 단계

이 스크립트로 만든 `results/*.jsonl`은 이후 ACC/TBR/Δ_AA 계산, leave-one-verb-out,
부트스트랩 신뢰구간 스크립트(별도 전달드린 `metrics.py` / `leave_one_verb_out.py` /
`bootstrap_ci.py`, 사용법은 `ANALYSIS_README.md` 참고)의 입력으로 그대로 사용하시면 됩니다.
