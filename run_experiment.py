"""
실행 예시
---------
파일럿(20문항, 두 모델 x 4전략 = 최대 160회 호출):
    python run_experiment.py --limit 20 --output results/pilot.jsonl

본실행(전체 400문항):
    python run_experiment.py --output results/main.jsonl

특정 모델/전략만:
    python run_experiment.py --models gpt --strategies zero-shot dap

중간에 끊겨도 같은 명령을 다시 실행하면 이미 완료된 (id, model, strategy)
조합은 건너뛰고 이어서 진행합니다 (output 파일을 읽어 자동 판단).
"""

import argparse
import json
import os
import time
from pathlib import Path

import config
import clients
from clients import CALL_FUNCS
from parser import parse_response
from prompts import PROMPT_TEMPLATES_KO


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_done_keys(output_path):
    """이미 '성공적으로' 결과가 저장된 (id, model, strategy) 조합만 완료로 간주.

    같은 키가 여러 번 나오면 파일에 쓰인 순서대로 마지막 상태를 따른다
    (즉, 실패 -> 재실행으로 성공 기록이 뒤에 덧붙으면 최종적으로 완료 처리됨).
    raw_response가 "ERROR:"로 시작하는 행은 API 호출 자체가 실패한 것이므로
    재시도 대상으로 남겨둔다 (완료 집합에서 제외).
    """
    status = {}  # key -> True(성공)/False(실패)
    if not os.path.exists(output_path):
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["id"], row["model"], row["strategy"])
                raw = row.get("raw_response") or ""
                succeeded = not str(raw).startswith("ERROR:")
                status[key] = succeeded  # 같은 키가 다시 나오면 마지막 값으로 덮어씀
            except (json.JSONDecodeError, KeyError):
                continue
    return {key for key, ok in status.items() if ok}


def run(dataset_path, output_path, models, strategies, limit=None, sleep_between=0.5):
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]

    done_keys = load_done_keys(output_path)
    print(f"기존 성공 완료 항목: {len(done_keys)}개 (실패/에러 기록은 재시도 대상에 포함됨)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_planned = len(dataset) * len(models) * len(strategies)
    done_count = 0
    fail_count = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for item in dataset:
            for strategy in strategies:
                template = PROMPT_TEMPLATES_KO[strategy]
                system_prompt = template["system"]
                user_prompt = template["user_format"].format(
                    premise=item["premise"], hypothesis=item["hypothesis"]
                )

                for model_key in models:
                    key = (item["id"], model_key, strategy)
                    if key in done_keys:
                        continue

                    call_fn = CALL_FUNCS[model_key]
                    try:
                        raw = call_fn(system_prompt, user_prompt)
                        parsed = parse_response(strategy, raw)
                    except Exception as e:  # noqa: BLE001
                        print(f"[실패] {item['id']} / {model_key} / {strategy}: {e}")
                        parsed = {"label": None, "raw": f"ERROR: {e}", "parse_ok": False}
                        fail_count += 1

                    record = {
                        "id": item["id"],
                        "group": item["group"],
                        "verb_class": item["verb_class"],
                        "verb": item["verb"],
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "gold_label": item["label"],
                        "model": model_key,
                        "model_name": config.MODELS[model_key],
                        "strategy": strategy,
                        "predicted_label": parsed.get("label"),
                        "parse_ok": parsed.get("parse_ok"),
                        "raw_response": parsed.get("raw"),
                        "reasoning": parsed.get("reasoning"),
                        "possible_interruptions": parsed.get("possible_interruptions"),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()  # 중간에 끊겨도 여기까지는 보존됨

                    done_count += 1
                    print(
                        f"[{len(done_keys) + done_count}/{total_planned}] "
                        f"{item['id']} / {model_key} / {strategy} -> {record['predicted_label']} "
                        f"(parse_ok={record['parse_ok']})"
                    )
                    time.sleep(sleep_between)

    print(f"\n완료: {output_path}")
    print(f"이번 실행에서 처리: {done_count}건, 실패(파싱 실패 포함 아님, 호출 에러): {fail_count}건")

    if "gemini" in models:
        clients.print_gemini_usage_summary()
    if "qwen" in models:
        clients.print_qwen_usage_summary()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="미완료상 역설 한국어 데이터셋 행동 실험 실행기")
    arg_parser.add_argument("--dataset", default=config.DATASET_PATH)
    arg_parser.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "results.jsonl"))
    arg_parser.add_argument(
        "--models", nargs="+", default=list(config.MODELS.keys()), choices=list(config.MODELS.keys())
    )
    arg_parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(PROMPT_TEMPLATES_KO.keys()),
        choices=list(PROMPT_TEMPLATES_KO.keys()),
    )
    arg_parser.add_argument("--limit", type=int, default=None, help="파일럿용: 데이터셋 앞에서 N개 문항만 실행")
    arg_parser.add_argument("--sleep", type=float, default=0.5, help="호출 사이 대기 시간(초), rate limit 방지용")
    args = arg_parser.parse_args()

    run(
        dataset_path=args.dataset,
        output_path=args.output,
        models=args.models,
        strategies=args.strategies,
        limit=args.limit,
        sleep_between=args.sleep,
    )
