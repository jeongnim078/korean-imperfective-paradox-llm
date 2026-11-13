"""
모델의 자유 텍스트 응답에서 참/거짓/모름 라벨을 뽑아내는 파서.
- zero-shot / dap: 순수 단어(또는 단어를 포함한 문장) 방어적으로 파싱
- cot / counterfactual: JSON 파싱 (마크다운 코드블록 제거 포함), 실패 시 텍스트 폴백
"""

import json
import re

LABEL_MAP = {"참": "True", "거짓": "False", "모름": "Unknown"}


def _strip_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_simple(raw_text):
    """zero-shot / dap용: 텍스트 안에 '참'/'거짓'/'모름' 중 무엇이 있는지 탐색.
    모델이 지시를 어기고 부연 설명을 붙이는 경우까지 방어."""
    text = (raw_text or "").strip()
    found = [ko for ko in LABEL_MAP if ko in text]
    if len(found) == 1:
        return {"label": LABEL_MAP[found[0]], "raw": raw_text, "parse_ok": True}
    # 여러 단어가 동시에 검출되면(예: 설명 중 다른 단어 언급) 애매하므로 실패 처리
    return {"label": None, "raw": raw_text, "parse_ok": False}


def parse_json_response(raw_text, expected_keys):
    cleaned = _strip_code_fence(raw_text or "")
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        fallback = parse_simple(raw_text)
        fallback["parse_ok"] = False
        for key in expected_keys:
            fallback[key] = None
        return fallback

    label_ko = obj.get("label")
    label_en = LABEL_MAP.get(label_ko.strip()) if isinstance(label_ko, str) else None
    result = {"label": label_en, "raw": raw_text, "parse_ok": label_en is not None}
    for key in expected_keys:
        result[key] = obj.get(key)
    return result


def parse_response(strategy, raw_text):
    if strategy in ("zero-shot", "dap"):
        return parse_simple(raw_text)
    if strategy == "cot":
        return parse_json_response(raw_text, expected_keys=["reasoning"])
    if strategy == "counterfactual":
        return parse_json_response(raw_text, expected_keys=["possible_interruptions"])
    raise ValueError(f"알 수 없는 전략입니다: {strategy}")
