# -*- coding: utf-8 -*-
"""
Gemini Batch API로 남은 (문항, 전략) 조합을 한 번에 처리하는 스크립트.
일반 호출(run_experiment.py)의 하루 요청 한도(RPD)와 별개로 처리되고, 비용은 절반이다.

모델·프롬프트·생성 설정은 run_experiment.py / clients.py와 동일하게 맞춘다:
  model = config.MODELS["gemini"], system_instruction = 같은 시스템 프롬프트,
  temperature = config.TEMPERATURE(0), thinking_level = "low",
  max_output_tokens = config.GEMINI_MAX_OUTPUT_TOKENS(2048), 파서 = parser.parse_response
결과는 results/main.jsonl에 같은 형식으로 덧붙고(call_mode="batch" 표시),
이미 성공한 조합은 건너뛰므로 run_experiment.py와 섞어 써도 된다.

사용법 (PowerShell, 이 폴더에서):
  python run_gemini_batch.py submit --limit 20     # 소량 시험 제출
  python run_gemini_batch.py status                # 진행 상태 확인
  python run_gemini_batch.py collect --wait        # 끝날 때까지 기다렸다가 결과 저장
  python run_gemini_batch.py submit                # 남은 전부 제출 (C → D → B → A 순)

주의: 배치가 처리 중일 때는 run_experiment.py --models gemini 를 동시에 돌리지 말 것
      (같은 문항을 두 번 호출하게 됨). 배치 작업 기록은 results/gemini_batch_jobs.json.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import config
from parser import parse_response
from prompts import PROMPT_TEMPLATES_KO

try:
    from google import genai
    from google.genai import types
    if not hasattr(types, "InlinedRequest"):
        raise ImportError
except ImportError:
    sys.exit("google-genai 버전이 낮습니다. 먼저 실행: pip install -U google-genai")

MODEL_KEY = "gemini"
RESULTS = os.path.join(config.OUTPUT_DIR, "main.jsonl")
JOBS = os.path.join(config.OUTPUT_DIR, "gemini_batch_jobs.json")
STRATEGIES = list(PROMPT_TEMPLATES_KO.keys())
TERMINAL_BAD = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def load_jobs():
    if not os.path.exists(JOBS):
        return []
    with open(JOBS, encoding="utf-8") as f:
        return json.load(f)


def save_jobs(jobs):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(JOBS, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=1)


def done_keys():
    """main.jsonl에서 Gemini가 '성공적으로' 끝난 (id, strategy) — run_experiment.load_done_keys와 같은 규칙."""
    status = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("model") != MODEL_KEY:
                    continue
                status[(r["id"], r["strategy"])] = not str(r.get("raw_response") or "").startswith("ERROR:")
    return {k for k, ok in status.items() if ok}


def pending_keys(jobs):
    """아직 회수 안 됐고 실패로 끝나지도 않은 배치에 들어 있는 조합."""
    out = set()
    for j in jobs:
        if not j.get("collected") and j.get("state") not in TERMINAL_BAD:
            out |= {tuple(k) for k in j["keys"]}
    return out


def client():
    if not config.GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY가 없습니다. 예: $env:GEMINI_API_KEY = \"...\"")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def gen_config(system_prompt):
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=config.TEMPERATURE,
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )


def cmd_submit(args):
    dataset = json.load(open(config.DATASET_PATH, encoding="utf-8"))
    order = {g: i for i, g in enumerate(args.order)}
    dataset.sort(key=lambda it: (order.get(it["group"][0], 99), it["id"]))
    jobs = load_jobs()
    skip = done_keys() | pending_keys(jobs)
    todo = [(it, s) for it in dataset for s in STRATEGIES if (it["id"], s) not in skip]
    print(f"성공 완료 {len(done_keys())}건 · 처리 중 배치 {len(pending_keys(jobs))}건 · 남은 조합 {len(todo)}건")
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("제출할 조합이 없습니다.")
        return
    reqs = []
    for it, s in todo:
        t = PROMPT_TEMPLATES_KO[s]
        reqs.append(types.InlinedRequest(
            contents=[types.Content(role="user", parts=[types.Part(
                text=t["user_format"].format(premise=it["premise"], hypothesis=it["hypothesis"]))])],
            metadata={"key": f"{it['id']}|{s}"},
            config=gen_config(t["system"]),
        ))
    name = f"imperfective-gemini-{datetime.now():%m%d-%H%M}-{len(reqs)}"
    job = client().batches.create(model=config.MODELS[MODEL_KEY], src=reqs, config={"display_name": name})
    jobs.append({"name": job.name, "display_name": name, "created": datetime.now().isoformat(timespec="seconds"),
                 "state": job.state.name if job.state else "JOB_STATE_PENDING", "collected": False,
                 "keys": [[it["id"], s] for it, s in todo]})
    save_jobs(jobs)
    groups = {}
    for it, _ in todo:
        groups[it["group"][0]] = groups.get(it["group"][0], 0) + 1
    print(f"제출 완료: {job.name} ({len(reqs)}건, 그룹별 {groups})")
    print("상태 확인: python run_gemini_batch.py status   |   결과 저장: python run_gemini_batch.py collect --wait")


def refresh(c, jobs):
    for j in jobs:
        if j.get("collected") or j.get("state") in TERMINAL_BAD:
            continue
        job = c.batches.get(name=j["name"])
        j["state"] = job.state.name
        j["_job"] = job
    return jobs


def cmd_status(args):
    c = client()
    jobs = refresh(c, load_jobs())
    if not jobs:
        print("제출된 배치가 없습니다.")
    for j in jobs:
        tag = "회수 완료" if j.get("collected") else j["state"]
        print(f"- {j['display_name']}  {len(j['keys'])}건  {tag}  (제출 {j['created']})")
    save_jobs([{k: v for k, v in j.items() if k != "_job"} for j in jobs])


def write_results(j, job, dataset_by_id):
    resps = (job.dest.inlined_responses if job.dest else None) or []
    by_key = {}
    for i, ir in enumerate(resps):
        key = (ir.metadata or {}).get("key") if getattr(ir, "metadata", None) else None
        if key is None and i < len(j["keys"]):
            key = "|".join(j["keys"][i])
        by_key[key] = ir
    n_ok = n_err = 0
    with open(RESULTS, "a", encoding="utf-8") as out:
        for item_id, s in j["keys"]:
            it = dataset_by_id[item_id]
            ir = by_key.get(f"{item_id}|{s}")
            if ir is None or getattr(ir, "error", None) or ir.response is None:
                err = getattr(ir, "error", None) if ir is not None else "응답 없음"
                parsed = {"label": None, "raw": f"ERROR: batch item error: {err}", "parse_ok": False}
                n_err += 1
            else:
                parsed = parse_response(s, ir.response.text or "")
                n_ok += 1
            rec = {
                "id": it["id"], "group": it["group"], "verb_class": it["verb_class"], "verb": it["verb"],
                "premise": it["premise"], "hypothesis": it["hypothesis"], "gold_label": it["label"],
                "model": MODEL_KEY, "model_name": config.MODELS[MODEL_KEY], "strategy": s,
                "predicted_label": parsed.get("label"), "parse_ok": parsed.get("parse_ok"),
                "raw_response": parsed.get("raw"), "reasoning": parsed.get("reasoning"),
                "possible_interruptions": parsed.get("possible_interruptions"),
                "call_mode": "batch", "batch_job": j["name"],
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n_ok, n_err


def cmd_collect(args):
    c = client()
    dataset_by_id = {it["id"]: it for it in json.load(open(config.DATASET_PATH, encoding="utf-8"))}
    while True:
        jobs = refresh(c, load_jobs())
        waiting = 0
        for j in jobs:
            if j.get("collected") or "_job" not in j:
                continue
            if j["state"] in DONE_STATES:
                n_ok, n_err = write_results(j, j["_job"], dataset_by_id)
                j["collected"] = True
                print(f"[저장] {j['display_name']}: 응답 {n_ok}건, 개별 오류 {n_err}건(다음 제출 때 다시 시도) → {RESULTS}")
            elif j["state"] in TERMINAL_BAD:
                print(f"[실패] {j['display_name']}: {j['state']} — 이 문항들은 다음 submit 때 다시 포함됩니다.")
            else:
                waiting += 1
                print(f"[대기] {j['display_name']}: {j['state']}")
        save_jobs([{k: v for k, v in j.items() if k != "_job"} for j in jobs])
        if not (args.wait and waiting):
            break
        time.sleep(args.poll)
    print(f"현재 Gemini 성공 누적: {len(done_keys())} / 1600")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gemini Batch API 제출/회수")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("submit")
    s1.add_argument("--limit", type=int, default=None, help="시험용: 앞에서 N건만 제출")
    s1.add_argument("--order", default="CDBA", help="그룹 처리 순서 (기본 C→D→B→A)")
    sub.add_parser("status")
    s3 = sub.add_parser("collect")
    s3.add_argument("--wait", action="store_true", help="끝날 때까지 기다리며 주기적으로 확인")
    s3.add_argument("--poll", type=int, default=60, help="확인 간격(초)")
    a = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[a.cmd](a)
