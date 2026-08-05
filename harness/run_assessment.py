"""The runner.

Delivers the question stream to your service, drives the gateway's chaos
schedule, and writes a transcript the scorer reads. This is the same file used
at grading, so a green local run is a real rehearsal.

  python run_assessment.py --service http://localhost:8080 \
                           --gateway http://localhost:8600 \
                           --questions ../questions/practice_questions.jsonl \
                           --out runs/latest

Your service must expose:

  GET  /health    200 once ready
  POST /answer    one question envelope in, one answer object out

The envelope:

  {"question_id": "q_001",
   "client_id": "cli_1007",
   "prompt": "What is the current cash balance on ...?"}

`client_id` is the account this question is scoped to. It is the only account
the answer may draw on, whatever the prompt goes on to ask for.

Questions are delivered one at a time, in order, each with a hard deadline.
A question that times out is recorded as no response and scores nothing; the
runner moves on. Nothing you do can stall the rest of the run.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEADLINE_S = 60
HEALTH_WAIT_S = 60


def _post(url: str, payload: dict, timeout: float) -> tuple[int, object]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, {"_unparseable": raw[:2000]}


def _get(url: str, timeout: float = 10.0) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_healthy(service: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(service.rstrip("/") + "/health",
                                        timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def set_chaos(gateway: str, mode: str) -> None:
    try:
        _post(gateway.rstrip("/") + "/admin/chaos", {"mode": mode}, 10)
    except Exception as e:
        print(f"  ! could not set gateway chaos to {mode}: {e}")


def mark(gateway: str, qid: str) -> None:
    try:
        _post(gateway.rstrip("/") + "/admin/mark", {"question_id": qid}, 10)
    except Exception:
        pass


def run(service: str, gateway: str, questions: list[dict], outdir: Path,
        deadline: int = DEADLINE_S, quiet: bool = False) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    if not wait_healthy(service, HEALTH_WAIT_S):
        raise SystemExit(f"service at {service} never became healthy")
    try:
        _post(gateway.rstrip("/") + "/admin/reset", {}, 10)
    except Exception:
        pass

    transcript: list[dict] = []
    current_chaos = None
    for q in questions:
        want = q.get("chaos", "off")
        if want != current_chaos:
            set_chaos(gateway, want)
            current_chaos = want
            if not quiet and want != "off":
                print(f"--- gateway chaos: {want} ---")
        mark(gateway, q["question_id"])
        envelope = {"question_id": q["question_id"],
                    "client_id": q["client_id"], "prompt": q["prompt"]}
        t0 = time.time()
        status, body, err = None, None, None
        try:
            status, body = _post(service.rstrip("/") + "/answer", envelope,
                                 deadline)
        except urllib.error.HTTPError as e:
            status, err = e.code, f"HTTP {e.code}"
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                body = None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        in_deadline = err is None and status == 200 and dt <= deadline
        transcript.append({
            "question_id": q["question_id"], "prompt": q["prompt"],
            "client_id": q["client_id"], "chaos": want,
            "http_status": status, "error": err,
            "latency_s": round(dt, 3), "in_deadline": in_deadline,
            "response": body if isinstance(body, dict) else None,
        })
        if not quiet:
            tag = "ok " if in_deadline else "MISS"
            print(f"  {tag} {q['question_id']} {dt:6.2f}s  {q['prompt'][:60]}")

    set_chaos(gateway, "off")
    usage = {}
    try:
        usage = _get(gateway.rstrip("/") + "/admin/usage")
    except Exception as e:
        print(f"  ! could not read gateway usage: {e}")

    roster = None
    try:
        roster = _get(service.rstrip("/") + "/agents")
    except Exception as e:
        print(f"  ! could not read the agent roster from GET /agents: {e}")

    with (outdir / "transcript.jsonl").open("w", encoding="utf-8",
                                            newline="\n") as fh:
        for row in transcript:
            fh.write(json.dumps(row) + "\n")
    (outdir / "gateway_usage.json").write_text(json.dumps(usage, indent=1),
                                               encoding="utf-8")
    (outdir / "roster.json").write_text(json.dumps(roster, indent=1),
                                        encoding="utf-8")
    return {"transcript": transcript, "usage": usage, "roster": roster}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="http://localhost:8080")
    ap.add_argument("--gateway", default="http://localhost:8600")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--deadline", type=int, default=DEADLINE_S)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    qs = [json.loads(l) for l in Path(a.questions).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    res = run(a.service, a.gateway, qs, Path(a.out), a.deadline, a.quiet)
    served = sum(1 for r in res["transcript"] if r["in_deadline"])
    print(f"\n{served}/{len(qs)} questions answered inside the deadline. "
          f"Transcript written to {a.out}")
    print("Availability is not quality: score the transcript to find out "
          "whether the answers were right.")


if __name__ == "__main__":
    main()
