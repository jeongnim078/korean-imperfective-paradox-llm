"""
실험 전역 설정.
API 키는 코드에 직접 쓰지 않고 환경변수로 읽습니다.

이 파일과 같은 폴더에 .env 파일을 만들어두면 자동으로 읽어옵니다:
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=...

.env 파일이 없다면 터미널에서 직접 환경변수를 설정해도 됩니다:
    (PowerShell) $env:OPENAI_API_KEY = "sk-..."
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()  # 같은 폴더의 .env 파일을 자동으로 찾아 읽음
except ImportError:
    pass  # python-dotenv 미설치 시 .env 자동 로드는 건너뛰고 환경변수만 사용

# --- 사용할 모델 ---
MODELS = {
    "gpt": "gpt-5.6-terra",
    "gemini": "gemini-3.1-pro-preview",
    # Qwen(오픈웨이트): 완전 로컬 구동(Ollama, CPU). API 비용이 전혀 들지 않음.
    #
    # 성능/로컬구동/비용 세 조건을 동시에 만족하는 모델은 없어서(플래그십급
    # 성능은 총 파라미터가 커서 CPU 로컬 서빙이 사실상 불가능함), 로컬 구동과
    # 비용을 우선하고 성능은 한 단계 아래인 모델을 선택함:
    #   Qwen3-30B-A3B-Instruct-2507 (MoE, 총 30.5B / 활성 3.3B)
    #   - "non-thinking" 전용 버전이라 <think> 블록을 만들지 않음 -> 응답 파싱이 단순함
    #   - 활성 파라미터가 3.3B뿐이라 총 파라미터(30B) 대비 CPU 추론이 상대적으로 빠름
    #   - Ollama q4_K_M 양자화 기준 파일 크기 약 19GB -> 32GB RAM에서 구동 가능
    # 한계: gemini-3.1-pro-preview / gpt-5.6-terra 같은 프론티어 모델과 동급은 아님.
    # (연구 한계로 명시할 것. 원 논문도 프로프라이어터리 vs 오픈모델 격차를 전제함)
    "qwen": "qwen3:30b-a3b-instruct-2507-q4_K_M",
}

# 메모리가 부족해 스왑이 발생하면(체감상 급격히 느려짐) 아래로 교체:
#   "qwen": "qwen3:30b-a3b-instruct-2507-q4_0"   # q4_K_M보다 약간 작고 품질은 소폭 하락
# RAM이 64GB 이상으로 늘어나면 성능을 더 올릴 수 있는 대안:
#   "qwen": "qwen3:30b-a3b-instruct-2507-q8_0"   # 품질 ↑, 파일 크기 ~33GB
#   "qwen": "Qwen/Qwen3-235B-A22B-Instruct-2507"  # 플래그십급, 단 API 호스팅 필요(유료, DeepInfra 등)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Qwen(오픈웨이트, 로컬 Ollama) 호출 설정 ---
# 준비:
#   1) https://ollama.com 에서 Ollama 설치
#   2) 터미널에서 모델 다운로드: ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
#   3) Ollama 서버 실행 확인 (보통 설치 시 자동 실행됨; 안 되어 있으면 `ollama serve`)
# Ollama는 OpenAI 호환 API를 기본 제공하므로(http://localhost:11434/v1),
# clients.py에서 openai SDK를 그대로 재사용한다. API 키는 로컬에서는 검증하지
# 않지만 openai SDK가 빈 문자열을 거부하므로 더미 값을 넣는다.
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:11434/v1")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "ollama")  # 로컬 서버는 실제 키 검증을 하지 않음

# CPU 로컬 추론은 상용 API보다 훨씬 느릴 수 있어 기본 타임아웃(REQUEST_TIMEOUT)과
# 별도로 넉넉하게 잡는다. 파일럿에서 문항 1~2개로 실제 응답 시간을 먼저 재보고
# 필요하면 이 값을 조정할 것.
QWEN_REQUEST_TIMEOUT = int(os.environ.get("QWEN_REQUEST_TIMEOUT", "300"))  # 초

# --- 경로 ---
DATASET_PATH = "imperfective_paradox_ko_full.json"
OUTPUT_DIR = "results"

# --- 호출 설정 ---
TEMPERATURE = 0.0  # 모델 간 공정한 비교를 위해 재현성 우선(모두 동일 값 사용)
MAX_RETRIES = 8  # 트래픽 폭증 상황을 감안해 5 -> 8로 상향
RATE_LIMIT_BASE_WAIT = 20  # 429/쿼터초과 감지 시 20,40,60...초로 선형 증가 대기
REQUEST_TIMEOUT = 60  # 초

# 일부 모델(주로 reasoning 계열)은 temperature 커스텀 값을 지원하지 않고
# 고정값만 허용한다. 예: gpt-5.6-terra는 temperature=0을 지정해도 실제로는
# 1로 적용됨(2026-08 파일럿에서 실측 확인 — API가 커스텀 값을 무시/거부함).
# 여기 등록해두면 clients.py가 해당 모델 호출 시 temperature 파라미터
# 자체를 보내지 않고(모델의 기본값을 그대로 사용), 실제 적용된 값을
# 콘솔에 한 번만 로그로 남긴다.
#
# 주의: 이 경우 gpt-5.6-terra는 TEMPERATURE=0.0으로 맞춘 다른 모델(gemini,
# qwen)과 달리 실제로는 temperature=1로 생성된다. 모델 간 완벽히 동일한
# 생성 조건은 아니라는 점을 논문/보고서 방법론(Methods) 절에 명시할 것.
FIXED_TEMPERATURE_MODELS = {
    "gpt-5.6-terra": 1.0,
}

# gemini-3.1-pro-preview는 thinking을 완전히 끌 수 없고(공식 문서 확인),
# thinking_level="low"로 낮춰도 GPT 대비 출력 토큰이 구조적으로 더 많이 나올 수
# 있다. max_output_tokens를 걸어 최악의 경우 비용을 상한선 안으로 묶는다.
# 너무 낮게 잡으면 thinking만 소비하고 최종 답변이 잘려 parse_ok=False가
# 늘어날 수 있으니, 파일럿 결과의 parse_ok 비율을 보고 조정할 것.
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
