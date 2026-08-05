"""Valura take-home LLM gateway.

An OpenAI-compatible chat-completions proxy. Your service must send every LLM
call here (the grading environment has no other network route). It exists so
that grading is fair and measurable:

  - It exposes exactly two model names: "valura-fast" and "valura-deep".
    Any other model name is rejected with 400. At grading time these map to
    pinned upstream snapshots; "valura-deep" is billed at 4x the tokens of
    "valura-fast".
  - It meters tokens per request and per question, which is how the cost
    dimension of your score is measured.
  - It injects upstream failures on a schedule. The grading run WILL include
    both bands, and this is the same file grading uses, so you can rehearse
    every failure mode locally.

      transient_429   the FIRST call for each question is rejected with 429
                      and a Retry-After header; every later call for that same
                      question succeeds. Retrying gets you through it.
      blackout        every call is rejected with a quota-exhausted error for
                      the whole band. Nothing gets you through it. The
                      question still has to be answered, or declined honestly.

Run it:

  python llm_gateway.py                          stub mode, no key needed
  UPSTREAM_MODE=passthrough \
  UPSTREAM_BASE_URL=https://api.openai.com/v1 \
  UPSTREAM_API_KEY=sk-... \
  MODEL_MAP_FAST=gpt-4.1-mini MODEL_MAP_DEEP=gpt-4.1 \
  python llm_gateway.py                          real upstream through the alias map

Endpoints:

  POST /v1/chat/completions      OpenAI-shaped; only the two alias models
  GET  /health
  POST /admin/chaos              {"mode": "off" | "transient_429" | "blackout"}
  POST /admin/mark               {"question_id": "..."} attribute usage to a question
  GET  /admin/usage              token and request accounting, per question
  POST /admin/reset              zero the meters and chaos state

Stub mode returns a fixed acknowledgement string and estimated token usage. It
exists to exercise plumbing, retries and failure handling without spending
anything; it does not understand your prompts. Develop the reasoning path
against a real upstream (your own key or a local OpenAI-compatible server).

Environment: GATEWAY_PORT (default 8600), UPSTREAM_MODE (stub|passthrough),
UPSTREAM_BASE_URL, UPSTREAM_API_KEY, MODEL_MAP_FAST, MODEL_MAP_DEEP,
TOKEN_CAP_BILLED (default 3000000; beyond it every request fails with 429
insufficient_quota, which is exactly what a blown budget does in production).

Standard library only. No auth: it is only ever reachable inside the isolated
assessment network.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALIASES = ("valura-fast", "valura-deep")
BILL_MULTIPLIER = {"valura-fast": 1, "valura-deep": 4}

_STATE_LOCK = threading.Lock()


class GatewayState:
    def __init__(self) -> None:
        self.chaos_mode = "off"            # off | transient_429 | blackout
        # In the transient band the FIRST upstream call for each question is
        # rejected with 429 + Retry-After, and every later call for that same
        # question succeeds. Rejecting on a request counter instead would let
        # a service with no retry logic through on roughly half the band by
        # luck, which measures nothing.
        self.transient_seen: set = set()
        self.question_id = None
        self.requests = 0
        self.rejected_429 = 0
        self.blackout_rejections = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.billed_tokens = 0
        self.by_question: dict[str, dict] = {}
        self.by_model: dict[str, dict] = {a: {"requests": 0, "tokens": 0} for a in ALIASES}

    def usage(self) -> dict:
        return {
            "requests": self.requests,
            "rejected_429": self.rejected_429,
            "blackout_rejections": self.blackout_rejections,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "billed_tokens": self.billed_tokens,
            "by_model": self.by_model,
            "by_question": self.by_question,
        }


STATE = GatewayState()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _record_usage(model: str, prompt_toks: int, completion_toks: int) -> None:
    with _STATE_LOCK:
        STATE.prompt_tokens += prompt_toks
        STATE.completion_tokens += completion_toks
        billed = (prompt_toks + completion_toks) * BILL_MULTIPLIER[model]
        STATE.billed_tokens += billed
        STATE.by_model[model]["requests"] += 1
        STATE.by_model[model]["tokens"] += prompt_toks + completion_toks
        qid = STATE.question_id or "_unattributed"
        q = STATE.by_question.setdefault(
            qid, {"requests": 0, "tokens": 0, "billed_tokens": 0,
                  "fast_requests": 0, "deep_requests": 0})
        q["requests"] += 1
        q["tokens"] += prompt_toks + completion_toks
        q["billed_tokens"] += billed
        # Per-model, per-question, because the assessment scores whether the
        # orchestrator spent a capable-tier call on a trivial question. That is
        # only checkable if the meter records which tier answered what.
        q["deep_requests" if model == "valura-deep" else "fast_requests"] += 1


def _stub_completion(body: dict, model: str) -> dict:
    prompt_text = json.dumps(body.get("messages", []))
    content = ("STUB-GATEWAY acknowledgement. This upstream does not reason; "
               "it exists so you can rehearse plumbing, retries and failure "
               "handling without spending tokens.")
    p, c = _estimate_tokens(prompt_text), _estimate_tokens(content)
    _record_usage(model, p, c)
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": p, "completion_tokens": c,
                  "total_tokens": p + c},
    }


def _passthrough_completion(body: dict, model: str) -> tuple[int, dict]:
    upstream_model = os.environ.get(
        "MODEL_MAP_FAST" if model == "valura-fast" else "MODEL_MAP_DEEP", "")
    if not upstream_model:
        return 500, {"error": {"message": "gateway missing MODEL_MAP_* config",
                               "type": "gateway_misconfigured"}}
    base = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
    payload = dict(body, model=upstream_model)
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("UPSTREAM_API_KEY", "")},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
        except Exception:
            detail = {"error": {"message": str(e), "type": "upstream_error"}}
        return e.code, detail
    except Exception as e:
        return 502, {"error": {"message": f"upstream unreachable: {e}",
                               "type": "upstream_unreachable"}}
    usage = data.get("usage") or {}
    p = int(usage.get("prompt_tokens") or 0)
    c = int(usage.get("completion_tokens") or 0)
    if p == 0 and c == 0:
        p = _estimate_tokens(json.dumps(body.get("messages", [])))
        c = _estimate_tokens(json.dumps(data.get("choices", "")))
    _record_usage(model, p, c)
    data["model"] = model
    return 200, data


class Handler(BaseHTTPRequestHandler):
    server_version = "ValuraGateway/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, obj: dict, extra_headers: dict | None = None) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "chaos": STATE.chaos_mode})
        elif self.path == "/admin/usage":
            with _STATE_LOCK:
                self._send(200, STATE.usage())
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        global STATE
        body = self._read_body()
        if self.path == "/admin/chaos":
            mode = body.get("mode", "off")
            if mode not in ("off", "transient_429", "blackout"):
                self._send(400, {"error": {"message": f"unknown chaos mode {mode!r}"}})
                return
            with _STATE_LOCK:
                STATE.chaos_mode = mode
                STATE.transient_seen = set()
            self._send(200, {"chaos": mode})
            return
        if self.path == "/admin/mark":
            with _STATE_LOCK:
                STATE.question_id = body.get("question_id")
            self._send(200, {"marked": STATE.question_id})
            return
        if self.path == "/admin/reset":
            with _STATE_LOCK:
                STATE = GatewayState()
            self._send(200, {"reset": True})
            return
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found"}})
            return

        model = body.get("model")
        if model not in ALIASES:
            self._send(400, {"error": {
                "message": f"model must be one of {list(ALIASES)}, got {model!r}",
                "type": "invalid_model"}})
            return

        cap = int(os.environ.get("TOKEN_CAP_BILLED", "3000000"))
        with _STATE_LOCK:
            over_cap = STATE.billed_tokens >= cap
            chaos = STATE.chaos_mode
            first_for_question = False
            if chaos == "transient_429":
                qid = STATE.question_id or "_unattributed"
                first_for_question = qid not in STATE.transient_seen
                STATE.transient_seen.add(qid)
        if over_cap:
            with _STATE_LOCK:
                STATE.requests += 1
                STATE.rejected_429 += 1
            self._send(429, {"error": {
                "message": "token budget for this run is exhausted",
                "type": "insufficient_quota"}})
            return
        if chaos == "blackout":
            with _STATE_LOCK:
                STATE.requests += 1
                STATE.blackout_rejections += 1
            self._send(429, {"error": {
                "message": "You exceeded your current quota. The upstream is "
                           "unavailable for the remainder of this outage.",
                "type": "insufficient_quota"}})
            return
        if chaos == "transient_429" and first_for_question:
            with _STATE_LOCK:
                STATE.requests += 1
                STATE.rejected_429 += 1
            self._send(429, {"error": {
                "message": "Rate limit reached. Retry after the indicated delay.",
                "type": "rate_limit_exceeded"}},
                extra_headers={"Retry-After": "1"})
            return

        with _STATE_LOCK:
            STATE.requests += 1
        if os.environ.get("UPSTREAM_MODE", "stub") == "passthrough":
            code, data = _passthrough_completion(body, model)
            self._send(code, data)
        else:
            self._send(200, _stub_completion(body, model))


def serve(port: int | None = None) -> ThreadingHTTPServer:
    port = port if port is not None else int(os.environ.get("GATEWAY_PORT", "8600"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    return httpd


if __name__ == "__main__":
    server = serve()
    print(f"valura llm gateway on :{server.server_address[1]} "
          f"(upstream={os.environ.get('UPSTREAM_MODE', 'stub')})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
