"""
모델 호출 래퍼. 모든 함수는 (system_prompt, user_prompt) -> raw_text(str) 시그니처로 통일.
호출 실패 시 지수 백오프(레이트리밋은 더 길게 선형 백오프)로 재시도.
"""

import json
import os
import random
import re
import time

from google import genai
from google.genai import types

import config


def _is_rate_limit_error(e):
    msg = str(e)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "rate limit" in msg.lower()
        or "quota" in msg.lower()
    )


# --- Gemini 일일 요청 한도 소진 전용 처리 ---
# generate_requests_per_model_per_day 같은 "일일" 쿼터는 초/분 단위 트래픽
# 폭증과 달리 짧게 기다린다고(혹은 8번 재시도한다고) 절대 풀리지 않는다.
# 이걸 그냥 일반 레이트리밋처럼 취급하면 남은 재시도 횟수를 전부 낭비하고,
# 이후 문항들에서도 매번 똑같은 에러를 받으려고 API를 계속 호출하게 된다.
# 그래서 별도 예외로 구분해서 (1) 재시도 없이 즉시 포기하고 (2) 한도가 풀릴
# 것으로 예상되는 시각까지는 API 호출 자체를 시도하지 않도록 한다.
#
# 본실행은 도중에 스크립트를 몇 번 재시작해야 할 수도 있으므로, 차단 시각을
# 메모리에만 두지 않고 파일에도 저장해둔다. 재시작 후에도 파일을 읽어 즉시
# 차단 상태를 복원하면, 이미 소진된 걸 알면서도 첫 호출을 API에 한 번 더
# 날려서 같은 에러를 받는 낭비를 막을 수 있다.
class GeminiDailyQuotaExhaustedError(RuntimeError):
    """Gemini 일일 요청 한도 소진. 재시도해도 소용없으므로 즉시 포기해야 함."""


_GEMINI_QUOTA_BLOCK_PATH = os.path.join(config.OUTPUT_DIR, "gemini_quota_block.json")


def _load_daily_quota_block():
    """이전 실행에서 저장해둔 차단 시각(epoch 초)을 파일에서 읽어온다.
    파일이 없거나 형식이 이상하면 조용히 None을 반환(차단 없음으로 간주)."""
    try:
        with open(_GEMINI_QUOTA_BLOCK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("blocked_until")
    except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError, TypeError):
        return None


def _save_daily_quota_block(blocked_until):
    """차단 시각을 파일에 저장한다. 스크립트가 재시작되어도 이 파일을 읽어
    바로 이어서 건너뛸 수 있게 하기 위함. 저장 실패는 치명적이지 않으므로
    경고만 출력하고 계속 진행한다(메모리상의 차단은 이번 실행 동안 유효)."""
    try:
        os.makedirs(os.path.dirname(_GEMINI_QUOTA_BLOCK_PATH) or ".", exist_ok=True)
        with open(_GEMINI_QUOTA_BLOCK_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "blocked_until": blocked_until,
                    "blocked_until_readable": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(blocked_until)
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except OSError as e:
        print(f"[Gemini] 차단 시각 파일 저장 실패(무시하고 계속 진행): {e}")


def _clear_daily_quota_block_file():
    """차단이 해제되었을 때 파일을 정리한다. 파일이 없어도 에러 없이 통과."""
    try:
        os.remove(_GEMINI_QUOTA_BLOCK_PATH)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"[Gemini] 차단 시각 파일 삭제 실패(무시): {e}")


# 모듈이 임포트되는 시점(=스크립트 시작 시점)에 이전 차단 기록을 즉시 복원한다.
_GEMINI_DAILY_QUOTA_BLOCKED_UNTIL = _load_daily_quota_block()
if _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL is not None:
    if _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL > time.time():
        _until_str = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(_GEMINI_DAILY_QUOTA_BLOCKED_UNTIL)
        )
        print(
            f"[Gemini] 이전 실행에서 저장된 일일 한도 차단 상태를 복원했습니다. "
            f"{_until_str}까지 Gemini 호출을 건너뜁니다(gpt/qwen은 영향 없음)."
        )
    else:
        # 저장된 차단 시각이 이미 지났으면 무시하고 파일도 정리해둔다.
        _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL = None
        _clear_daily_quota_block_file()


def _is_daily_quota_exhausted_error(e):
    msg = str(e)
    # 예: "Quota exceeded for metric: ...generate_requests_per_model_per_day..."
    return "RESOURCE_EXHAUSTED" in msg and "per_day" in msg


def _extract_retry_delay_seconds(e):
    """에러 메시지에서 'retryDelay': '78019s' 형태를 찾아 초 단위 정수로 반환.
    못 찾으면 None (호출부에서 안전한 기본값으로 대체함)."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+)s", str(e))
    return int(m.group(1)) if m else None


def _retry_wrapper(label, fn):
    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return fn()
        except GeminiDailyQuotaExhaustedError:
            # 일일 한도 소진은 몇 초/몇 분 기다린다고 풀리는 문제가 아니므로
            # 재시도 루프를 돌지 않고 즉시 상위(run_experiment.py)로 전파한다.
            raise
        except Exception as e:  # noqa: BLE001 - 실험 스크립트이므로 광범위하게 재시도
            last_err = e
            if _is_rate_limit_error(e):
                # 트래픽 폭증/쿼터 초과는 몇 초 기다린다고 안 풀리는 경우가 많아서
                # 훨씬 길게, 선형으로 늘려가며 기다림 (예: 20,40,60,...초)
                wait = config.RATE_LIMIT_BASE_WAIT * (attempt + 1)
                print(
                    f"[{label}] 트래픽 폭증/한도 초과 감지, {wait}초 후 재시도 "
                    f"({attempt + 1}/{config.MAX_RETRIES}): {e}"
                )
            else:
                wait = (2**attempt) + random.random()
                print(f"[{label}] 오류 발생, {wait:.1f}초 후 재시도 ({attempt + 1}/{config.MAX_RETRIES}): {e}")
            time.sleep(wait)
    raise RuntimeError(f"{label} 호출이 재시도 한도({config.MAX_RETRIES}회)를 초과했습니다: {last_err}")


# --- OpenAI ---
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        if not config.OPENAI_API_KEY:
            raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=config.REQUEST_TIMEOUT)
    return _openai_client


_TEMPERATURE_WARNED_MODELS = set()  # 모델별로 경고를 한 번만 출력하기 위한 집합


def call_openai(system_prompt, user_prompt, model=None, temperature=None):
    model = model or config.MODELS["gpt"]
    requested_temperature = config.TEMPERATURE if temperature is None else temperature
    client = _get_openai_client()

    # config.FIXED_TEMPERATURE_MODELS에 등록된 모델(예: gpt-5.6-terra)은
    # temperature 파라미터를 아예 보내지 않고 모델 기본값을 그대로 쓴다.
    fixed_temp = config.FIXED_TEMPERATURE_MODELS.get(model)
    send_temperature = fixed_temp is None

    if fixed_temp is not None and model not in _TEMPERATURE_WARNED_MODELS:
        print(
            f"[OpenAI] 모델 '{model}'은 temperature 커스텀을 지원하지 않아 "
            f"고정값({fixed_temp})으로 생성됩니다(요청값 {requested_temperature} 무시됨). "
            f"config.FIXED_TEMPERATURE_MODELS에 등록되어 있어 자동으로 파라미터를 생략합니다."
        )
        _TEMPERATURE_WARNED_MODELS.add(model)

    def _build_kwargs(include_temperature):
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if include_temperature:
            kwargs["temperature"] = requested_temperature
        return kwargs

    def _do():
        try:
            resp = client.chat.completions.create(**_build_kwargs(send_temperature))
        except Exception as e:  # noqa: BLE001
            if _is_rate_limit_error(e):
                raise  # 레이트리밋은 여기서 처리하지 않고 _retry_wrapper로 넘김
            # config에 등록 안 된 모델인데 API가 "temperature 미지원" 오류를
            # 내는 경우, 한 번 감지해서 temperature 없이 자동 재시도한다.
            # (등록해두면 다음 실행부터는 이 분기를 타지 않고 바로 건너뜀.)
            msg = str(e).lower()
            looks_like_temp_error = "temperature" in msg and (
                "unsupported" in msg or "does not support" in msg or "not supported" in msg
            )
            if send_temperature and looks_like_temp_error:
                if model not in _TEMPERATURE_WARNED_MODELS:
                    print(
                        f"[OpenAI] 모델 '{model}'이 temperature={requested_temperature}를 거부했습니다: {e}\n"
                        f"         temperature 없이 재시도하고, config.FIXED_TEMPERATURE_MODELS에 "
                        f"이 모델을 등록해두는 것을 권장합니다."
                    )
                    _TEMPERATURE_WARNED_MODELS.add(model)
                resp = client.chat.completions.create(**_build_kwargs(False))
            else:
                raise
        return resp.choices[0].message.content

    return _retry_wrapper("OpenAI", _do)


# --- Gemini ---
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("환경변수 GEMINI_API_KEY가 설정되어 있지 않습니다.")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


# thinking_level=low 적용 후 실제로 사고 토큰이 줄었는지 파일럿에서 바로 확인하기 위한 누적 집계
GEMINI_USAGE_LOG = {"calls": 0, "thought_tokens": 0, "output_tokens": 0}
_GEMINI_THINKING_WARNED = False  # thinking_config 관련 경고를 한 번만 출력하기 위한 플래그


def call_gemini(system_prompt, user_prompt, model=None, temperature=None):
    global _GEMINI_THINKING_WARNED, _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL
    model = model or config.MODELS["gemini"]
    temperature = config.TEMPERATURE if temperature is None else temperature

    # 이전 호출에서 일일 한도 소진이 감지된 상태라면, 한도가 풀릴 것으로
    # 예상되는 시각이 지나기 전까지는 API 호출 자체를 시도하지 않고 즉시
    # 실패 처리한다. gpt/qwen 호출 경로는 건드리지 않으므로 영향 없음.
    if _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL is not None:
        now = time.time()
        if now < _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL:
            remaining_h = (_GEMINI_DAILY_QUOTA_BLOCKED_UNTIL - now) / 3600
            until_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_GEMINI_DAILY_QUOTA_BLOCKED_UNTIL))
            raise GeminiDailyQuotaExhaustedError(
                f"Gemini 일일 요청 한도 소진 상태라 호출을 건너뜁니다. "
                f"약 {remaining_h:.1f}시간 후({until_str}) 재개 예정."
            )
        # 예상 한도 초기화 시각이 지났으므로 차단을 해제하고 다시 시도해본다.
        print(f"[Gemini] 예상 한도 초기화 시각이 지나 호출을 재개합니다.")
        _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL = None
        _clear_daily_quota_block_file()

    client = _get_gemini_client()

    def _build_config(include_thinking):
        kwargs = dict(
            system_instruction=system_prompt,
            temperature=temperature,
            # 출력 토큰 상한선(안전장치). Gemini 3.x는 thinking을 완전히 끌 수 없어
            # thinking 토큰도 이 한도 안에서 소비된다. 너무 낮게 잡으면 답변이
            # 잘려서 parse_ok=False가 급증할 수 있으니, 파일럿에서 parse_ok 비율을
            # 보고 필요하면 config.GEMINI_MAX_OUTPUT_TOKENS 값을 조정할 것.
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        )
        if include_thinking:
            # 중요: thinking_level은 GenerateContentConfig 최상위가 아니라
            # thinking_config 안에 중첩되어야 한다. (config={"thinking_level": ...}
            # 처럼 최상위에 두면 "Extra inputs are not permitted" 오류가 남 —
            # 실제로 겪었던 에러의 원인이 SDK 구버전이 아니라 이 필드 위치였을
            # 가능성이 높음.)
            # gemini-3.1-pro-preview는 사고를 완전히 끌 수 없고 최저 단계가 "낮음"임.
            # 4개 프롬프트 전략 간 비교가 무의미해지는 것을 막기 위해 최저로 고정.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        return types.GenerateContentConfig(**kwargs)

    def _do():
        global _GEMINI_THINKING_WARNED, _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL
        try:
            resp = client.models.generate_content(
                model=model, contents=user_prompt, config=_build_config(True)
            )
        except Exception as e:  # noqa: BLE001
            if _is_daily_quota_exhausted_error(e):
                delay = _extract_retry_delay_seconds(e)
                if delay is None:
                    # retryDelay 파싱에 실패하면 안전하게 24시간 차단으로 대체
                    delay = 24 * 3600
                    print(
                        "[Gemini] 일일 요청 한도 소진 감지했지만 retryDelay 파싱에 실패했습니다. "
                        "안전하게 24시간 동안 호출을 건너뜁니다."
                    )
                _GEMINI_DAILY_QUOTA_BLOCKED_UNTIL = time.time() + delay
                _save_daily_quota_block(_GEMINI_DAILY_QUOTA_BLOCKED_UNTIL)
                until_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(_GEMINI_DAILY_QUOTA_BLOCKED_UNTIL)
                )
                print(
                    f"[Gemini] 일일 요청 한도 소진 감지. 재시도하지 않고 즉시 포기하며, "
                    f"{until_str}까지({delay/3600:.1f}시간) 이후 모든 Gemini 호출을 건너뜁니다. "
                    f"gpt/qwen 호출은 계속 진행됩니다. (차단 상태는 {_GEMINI_QUOTA_BLOCK_PATH}에 "
                    f"저장되어 스크립트를 재시작해도 유지됩니다.)"
                )
                raise GeminiDailyQuotaExhaustedError(str(e)) from e
            if _is_rate_limit_error(e):
                raise  # 레이트리밋은 여기서 처리하지 않고 _retry_wrapper로 넘김
            # 필드 위치를 고쳐도 혹시 설치된 google-genai 버전이 thinking_level
            # 자체를 아직 모르는 경우(구버전) 대비 -> thinking_config 없이 한 번 더
            # 시도한다. (가능하면 `pip install -U google-genai`로 업그레이드 권장)
            msg = str(e).lower()
            looks_like_schema_error = "thinking" in msg and ("extra_forbidden" in msg or "extra inputs" in msg)
            if looks_like_schema_error:
                if not _GEMINI_THINKING_WARNED:
                    print(
                        f"[Gemini] thinking_level 관련 스키마 오류가 발생했습니다: {e}\n"
                        f"         thinking_config 없이(기본 동적 사고) 재시도합니다. "
                        f"`pip install -U google-genai`로 업그레이드를 권장합니다."
                    )
                    _GEMINI_THINKING_WARNED = True
                resp = client.models.generate_content(
                    model=model, contents=user_prompt, config=_build_config(False)
                )
            else:
                raise

        usage = getattr(resp, "usage_metadata", None)
        thought_tokens = getattr(usage, "thoughts_token_count", None) or 0
        output_tokens = getattr(usage, "candidates_token_count", None) or 0
        GEMINI_USAGE_LOG["calls"] += 1
        GEMINI_USAGE_LOG["thought_tokens"] += thought_tokens
        GEMINI_USAGE_LOG["output_tokens"] += output_tokens
        print(f"    [Gemini usage] thought_tokens={thought_tokens}, output_tokens={output_tokens}")

        if not resp.text:
            # max_output_tokens 한도 안에서 thinking만 다 쓰고 최종 답변이
            # 잘렸을 가능성 -> 에러를 내지 않고 빈 문자열을 반환해 재시도
            # 낭비를 막는다. parser.py에서 parse_ok=False로 기록됨.
            print(
                "    [Gemini] 응답 텍스트가 비어 있습니다. max_output_tokens 한도 안에서 "
                "thinking만 소비하고 답변이 잘렸을 수 있습니다. "
                "config.GEMINI_MAX_OUTPUT_TOKENS를 늘리는 것을 고려하세요."
            )
            return ""

        return resp.text

    return _retry_wrapper("Gemini", _do)


def print_gemini_usage_summary():
    n = GEMINI_USAGE_LOG["calls"]
    if n == 0:
        print("Gemini 호출 기록 없음")
        return
    print(
        f"\n[Gemini 사용량 요약] 호출 {n}회 | "
        f"사고 토큰 합계 {GEMINI_USAGE_LOG['thought_tokens']} (평균 {GEMINI_USAGE_LOG['thought_tokens']/n:.1f}) | "
        f"출력 토큰 합계 {GEMINI_USAGE_LOG['output_tokens']} (평균 {GEMINI_USAGE_LOG['output_tokens']/n:.1f})"
    )


# --- Qwen (오픈웨이트, 로컬 Ollama 구동) ---
# Ollama가 기본 제공하는 OpenAI 호환 엔드포인트(http://localhost:11434/v1)를
# openai SDK로 그대로 호출한다. call_openai()와 코드 구조는 거의 동일하되,
# - base_url이 로컬 주소이고
# - 타임아웃을 CPU 추론에 맞게 더 길게 잡는다(config.QWEN_REQUEST_TIMEOUT).
# Qwen3-30B-A3B-Instruct-2507은 non-thinking 전용 버전이라 <think> 블록을
# 만들지 않으므로, Gemini와 달리 별도 사고과정 분리 로직이 필요 없다 —
# message.content가 곧 최종 답변이다.
_qwen_client = None


def _get_qwen_client():
    global _qwen_client
    if _qwen_client is None:
        from openai import OpenAI

        _qwen_client = OpenAI(
            api_key=config.QWEN_API_KEY,  # 로컬 서버는 실제 키 검증을 하지 않음(더미 값)
            base_url=config.QWEN_BASE_URL,
            timeout=config.QWEN_REQUEST_TIMEOUT,
        )
    return _qwen_client


# 비용은 없지만, CPU 로컬 추론 속도를 체감하기 위한 누적 집계
# (파일럿 단계에서 "문항당 몇 초 걸리는지"를 보고 전체 실행 시간을 가늠하는 용도).
QWEN_USAGE_LOG = {"calls": 0, "output_tokens": 0, "total_seconds": 0.0}


def call_qwen(system_prompt, user_prompt, model=None, temperature=None):
    model = model or config.MODELS["qwen"]
    temperature = config.TEMPERATURE if temperature is None else temperature
    client = _get_qwen_client()

    def _do():
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        elapsed = time.time() - t0

        content = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) or 0

        QWEN_USAGE_LOG["calls"] += 1
        QWEN_USAGE_LOG["output_tokens"] += completion_tokens
        QWEN_USAGE_LOG["total_seconds"] += elapsed
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0
        print(f"    [Qwen 로컬] {elapsed:.1f}초, {completion_tokens} 토큰 (~{tps:.1f} tok/s)")

        if not content or not content.strip():
            raise RuntimeError("Qwen 응답이 비어 있습니다(Ollama 서버가 켜져 있는지 확인하세요).")

        return content

    return _retry_wrapper("Qwen(로컬)", _do)


def print_qwen_usage_summary():
    n = QWEN_USAGE_LOG["calls"]
    if n == 0:
        print("Qwen 호출 기록 없음")
        return
    total_s = QWEN_USAGE_LOG["total_seconds"]
    print(
        f"\n[Qwen 로컬 사용량 요약] 호출 {n}회 | "
        f"총 소요 {total_s/60:.1f}분 (평균 {total_s/n:.1f}초/호출) | "
        f"출력 토큰 합계 {QWEN_USAGE_LOG['output_tokens']} "
        f"(평균 처리량 ~{QWEN_USAGE_LOG['output_tokens']/total_s:.1f} tok/s)"
    )
    remaining_hint = total_s / n
    print(
        f"[안내] 본실행(400문항 x 4전략 = 1,600회, Qwen만 기준)은 "
        f"위 평균 기준 약 {remaining_hint * 1600 / 3600:.1f}시간 소요 예상 "
        f"(파일럿 규모로 추정한 값이므로 실제와 다를 수 있음)."
    )


# --- 모델 키 -> 호출 함수 매핑 ---
CALL_FUNCS = {
    "gpt": call_openai,
    "gemini": call_gemini,
    "qwen": call_qwen,
}
