"""
API 키 없이도 돌려볼 수 있는 파서 단위 테스트.
    python test_parser.py
"""

from parser import parse_response

CASES = [
    # (전략, 모델 원문 응답 예시, 기대 라벨)
    ("zero-shot", "참", "True"),
    ("zero-shot", "거짓", "False"),
    ("zero-shot", "모름", "Unknown"),
    ("zero-shot", '"참"', "True"),  # 따옴표 붙여 답하는 경우
    ("dap", "정답: 거짓", "False"),  # 지시를 어기고 접두어를 붙인 경우
    (
        "cot",
        '{\n  "reasoning": "완결동사이므로...",\n  "label": "거짓"\n}',
        "False",
    ),
    (
        "cot",
        '```json\n{"reasoning": "...", "label": "참"}\n```',  # 마크다운 코드블록 포함
        "True",
    ),
    (
        "counterfactual",
        '{"possible_interruptions": ["a", "b", "c"], "label": "모름"}',
        "Unknown",
    ),
    ("cot", "이건 JSON이 아닌 그냥 텍스트인데 참 이라고 생각합니다", "True"),  # JSON 파싱 실패 폴백
]

if __name__ == "__main__":
    ok = 0
    for strategy, raw, expected in CASES:
        result = parse_response(strategy, raw)
        status = "OK" if result["label"] == expected else "FAIL"
        if status == "OK":
            ok += 1
        print(f"[{status}] strategy={strategy!r} raw={raw[:40]!r}... -> {result['label']} (기대: {expected})")

    print(f"\n{ok}/{len(CASES)} 통과")
