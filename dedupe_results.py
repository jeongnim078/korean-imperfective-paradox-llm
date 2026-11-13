"""
재시작으로 인해 같은 (id, model, strategy)가 여러 번 기록된 경우
마지막 성공 기록만 남기고 나머지(주로 앞선 ERROR 기록)를 제거.
metrics.py 등 분석 스크립트를 돌리기 전에 한 번 실행하세요.

사용법:
    python dedupe_results.py --input results/main.jsonl --output results/main_clean.jsonl
"""

import argparse
import json


def dedupe(input_path, output_path):
    latest = {}  # key -> row (마지막에 읽은 것이 남음)
    order = []   # 최초 등장 순서 보존용

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["id"], row["model"], row["strategy"])
            if key not in latest:
                order.append(key)
            latest[key] = row  # 나중에 나온 행으로 덮어씀 (재시도 성공분이 우선됨)

    error_count = sum(1 for row in latest.values() if str(row.get("raw_response", "")).startswith("ERROR:"))

    with open(output_path, "w", encoding="utf-8") as f:
        for key in order:
            f.write(json.dumps(latest[key], ensure_ascii=False) + "\n")

    print(f"입력 줄 수 처리 완료 -> 고유 키 {len(order)}개")
    print(f"여전히 ERROR로 남은 항목: {error_count}개 (있다면 --limit으로 추가 재시도 필요)")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dedupe(args.input, args.output)
