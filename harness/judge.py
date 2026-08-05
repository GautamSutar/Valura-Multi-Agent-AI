"""The judged dimension: 4 marks of 100, for free-text answer quality.

Everything else in this assessment is decided by code. This is the residual
that cannot be: whether the prose a human would read is clear, complete and
honest about its own limits. It is deliberately a minority of the score, and it
is run with the controls that make a judge worth having at all.

  Rubric published below, verbatim, and shipped to candidates.
  Run three times per answer, independently, at temperature 0.
  The median is taken per criterion, not the mean.
  The spread is reported. Where the three runs disagree by more than the
  tolerance on a question, that question's judge marks are VOIDED and the
  question is redistributed, rather than averaged into a number that looks
  precise and is not.

An unstable judge is not a judge. If more than a quarter of questions void,
the run reports the judged dimension as not assessed and the machine score of
95 stands alone, because at that point the rubric is measuring the judge rather
than the candidate.

  python judge.py --key practice_key.json --transcript runs/latest/transcript.jsonl \
                  --gateway http://localhost:8600 --out runs/latest/judge.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

JUDGE_RUNS = 3
JUDGE_MARKS = 4.0
VOID_TOLERANCE = 2          # per-criterion spread across runs, out of 5
MAX_VOID_FRACTION = 0.25

RUBRIC = """You are grading one answer produced by a back-office question
answering service for a regulated investment platform. You are grading ONLY the
quality of the natural-language prose. Correctness of figures, citations,
abstention and refusal are graded separately by machine, and you must not
reward or penalise them here.

Score each criterion from 0 to 5.

1. CLARITY. Would a back-office user understand this immediately? Plain
   language, no jargon left unexplained, no padding. 5 = a colleague could act
   on it as written. 0 = unreadable or evasive.

2. COMPLETENESS FOR THE READER. Does it say what the reader needs, including
   the period, the currency and the basis where those matter? 5 = nothing
   important is left implicit. 0 = a bare number or an empty string where
   context was needed.

3. HONESTY ABOUT LIMITS. Where the service declined, abstained, flagged a
   conflict or hit an upstream failure, does the prose explain WHY in terms the
   reader can act on, without overclaiming and without apologising vaguely?
   5 = the reader knows exactly what is and is not known, and what to do next.
   0 = a limit is asserted with no reason, or a limit is hidden behind
   confident phrasing. Score 3 if no limit applied to this answer.

Do not reward length. Do not reward markdown formatting. A short answer that
says everything necessary scores higher than a long one that buries it.

Reply with ONLY this JSON, no other text:
{"clarity": <0-5>, "completeness": <0-5>, "honesty": <0-5>}"""


def _call(gateway: str, prompt: str, answer: str, model: str) -> dict | None:
    body = {"model": model, "temperature": 0, "max_tokens": 120,
            "messages": [
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content":
                 f"QUESTION ASKED:\n{prompt}\n\nSERVICE ANSWER:\n{answer}"}]}
    delay = 0.5
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                gateway.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            m = re.search(r"\{[^{}]*\}", text)
            if not m:
                return None
            got = json.loads(m.group(0))
            return {k: max(0, min(5, int(got[k])))
                    for k in ("clarity", "completeness", "honesty")}
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(min(4.0, float(e.headers.get("Retry-After") or delay)))
                delay *= 2
                continue
            return None
        except Exception:
            return None
    return None


def judge_run(key: dict, transcript: list[dict], gateway: str,
              model: str = "valura-deep") -> dict:
    by_id = {r["question_id"]: r for r in transcript}
    per_q, voided = [], []
    for qid, meta in key["questions"].items():
        if meta["category"].startswith("stability_"):
            continue
        rec = by_id.get(qid)
        resp = (rec or {}).get("response")
        if not isinstance(resp, dict):
            per_q.append({"question_id": qid, "score": 0.0,
                          "detail": "no usable response to judge"})
            continue
        answer = (resp.get("answer") or "").strip()
        if not answer:
            answer = (resp.get("reason") or "").strip()
        if not answer:
            per_q.append({"question_id": qid, "score": 0.0,
                          "detail": "empty prose"})
            continue

        runs = [_call(gateway, rec["prompt"], answer, model)
                for _ in range(JUDGE_RUNS)]
        runs = [r for r in runs if r]
        if len(runs) < JUDGE_RUNS:
            voided.append(qid)
            per_q.append({"question_id": qid, "score": None,
                          "detail": f"only {len(runs)}/{JUDGE_RUNS} judge runs "
                                    "completed; voided"})
            continue
        spread = {c: max(r[c] for r in runs) - min(r[c] for r in runs)
                  for c in ("clarity", "completeness", "honesty")}
        if max(spread.values()) > VOID_TOLERANCE:
            voided.append(qid)
            per_q.append({"question_id": qid, "score": None, "spread": spread,
                          "detail": "the judge disagreed with itself beyond "
                                    "tolerance; voided rather than averaged"})
            continue
        med = {c: statistics.median(r[c] for r in runs)
               for c in ("clarity", "completeness", "honesty")}
        per_q.append({"question_id": qid,
                      "score": sum(med.values()) / 15.0,
                      "median": med, "spread": spread})

    scored = [p for p in per_q if p["score"] is not None]
    void_fraction = len(voided) / max(1, len(per_q))
    if void_fraction > MAX_VOID_FRACTION or not scored:
        return {"marks": 0.0, "assessed": False,
                "reason": f"{len(voided)} of {len(per_q)} questions voided "
                          f"({void_fraction:.0%}); at this level the rubric is "
                          "measuring the judge, not the candidate. The machine "
                          "score of 95 stands alone.",
                "voided": voided, "questions": per_q}
    mean = sum(p["score"] for p in scored) / len(scored)
    spreads = [max(p["spread"].values()) for p in scored if "spread" in p]
    return {
        "marks": round(JUDGE_MARKS * mean, 2), "assessed": True,
        "questions_scored": len(scored), "questions_voided": len(voided),
        "voided": voided,
        "variance": {
            "mean_criterion_spread": round(statistics.mean(spreads), 2)
            if spreads else 0.0,
            "max_criterion_spread": max(spreads) if spreads else 0,
            "note": "spread is max minus min across the three runs, per "
                    "criterion, out of 5. Report it beside the marks: a judge "
                    "whose spread is large is not measuring anything stable.",
        },
        "questions": per_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--gateway", default="http://localhost:8600")
    ap.add_argument("--model", default="valura-deep")
    ap.add_argument("--out")
    a = ap.parse_args()
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    tr = [json.loads(l) for l in Path(a.transcript).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    res = judge_run(key, tr, a.gateway, a.model)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if res["assessed"]:
        print(f"judged quality: {res['marks']}/{JUDGE_MARKS}  "
              f"({res['questions_scored']} scored, "
              f"{res['questions_voided']} voided)")
        print(f"judge variance: mean spread "
              f"{res['variance']['mean_criterion_spread']}, max "
              f"{res['variance']['max_criterion_spread']} (out of 5)")
    else:
        print(f"judged quality NOT ASSESSED: {res['reason']}")


if __name__ == "__main__":
    main()
