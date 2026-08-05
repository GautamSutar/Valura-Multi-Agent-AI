"""Reference client.

Doubles as the worked example candidates receive: it shows the whole protocol
loop end to end, in about eighty lines, with the retry and resume behaviour
the assessment expects. What it does NOT show is how to answer the questions,
which is the entire exercise.

  python -m arena.client --url https://… --key vlr_… --mode practice
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, url: str, key: str, mode: str = "practice"):
        self.url = url.rstrip("/")
        self.key = key
        self.mode = mode

    def _call(self, method: str, path: str, body=None, retries: int = 5):
        sep = "&" if "?" in path else "?"
        full = f"{self.url}{path}{sep}mode={self.mode}"
        delay = 0.5
        for attempt in range(retries):
            req = urllib.request.Request(
                full,
                data=json.dumps(body).encode("utf-8") if body is not None else None,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.key}"},
                method=method)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                # 429 is expected: the assessment injects rate limits and a
                # quota blackout. Back off and try again; do not give up on
                # the run because one call failed.
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(min(4.0, float(e.headers.get("Retry-After")
                                              or delay)))
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code} {path}: {raw[:300]}")
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"{type(e).__name__} on {path}: {e}")

    # -- protocol -----------------------------------------------------------
    def rules(self):
        return self._call("GET", "/v1/rules")

    def book(self):
        return self._call("GET", "/v1/book")

    def market(self):
        return self._call("GET", "/v1/market")

    def declare_roster(self, roster: dict):
        return self._call("POST", "/v1/roster", roster)

    def next_question(self):
        return self._call("GET", "/v1/next")

    def submit(self, answer: dict):
        return self._call("POST", "/v1/answer", answer)

    def me(self):
        return self._call("GET", "/v1/me")

    def llm_base_url(self) -> str:
        return f"{self.url}/llm/v1"


def run(url: str, key: str, mode: str, answerer, quiet: bool = False) -> dict:
    """The whole loop. `answerer` is called with (book, question) and returns
    the answer object. Everything else here is protocol."""
    c = Client(url, key, mode)
    book = c.book()
    if hasattr(answerer, "roster"):
        c.declare_roster(answerer.roster())
    n = 0
    while True:
        q = c.next_question()
        if q.get("done"):
            break
        try:
            answer = answerer.answer(q["question_id"], q["client_id"],
                                     q["prompt"])
        except Exception as e:
            # Never abandon the run because one question raised. Submit
            # something well-formed that says so; a missing answer scores the
            # same as a wrong one, and a crashed client scores nothing at all
            # for everything after it.
            answer = {"question_id": q["question_id"], "answer": "",
                      "answer_value": None, "abstained": True,
                      "refused": False,
                      "reason": f"internal error: {type(e).__name__}",
                      "citations": [], "confidence": 0.0, "flags": [],
                      "agents": ["router"]}
        res = c.submit(answer)
        n += 1
        if not quiet:
            tag = "ok " if res.get("in_deadline") else "LATE"
            print(f"  {tag} {q['question_id']} {res.get('latency_s')}s "
                  f"{q['prompt'][:56]}")
            fb = res.get("feedback")
            if fb and fb["marks"] < fb["marks_available"]:
                print(f"       {fb['marks']}/{fb['marks_available']} "
                      f"{fb['notes']}")
    return c.me()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--mode", default="practice",
                    choices=["practice", "graded"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    from takehome.ref_service import Reference
    c = Client(a.url, a.key, a.mode)
    ref = Reference(c.book(), c.llm_base_url(), api_key=a.key,
                    market=c.market())
    me = run(a.url, a.key, a.mode, ref, a.quiet)
    print(json.dumps({k: v for k, v in me.items() if k != "scorecard"},
                     indent=1))
    if me.get("scorecard"):
        print(S_render(me["scorecard"]))


def S_render(card):
    import score as S
    return S.render(card)


if __name__ == "__main__":
    main()
