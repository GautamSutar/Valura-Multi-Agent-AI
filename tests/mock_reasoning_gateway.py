"""A tiny stand-in for a real reasoning model, used only in local tests.

The bundled gateway's stub mode returns a fixed acknowledgement string and
cannot exercise structured output or tool-style narration, so the
orchestrator's LLM-dependent paths (router classification, notes narration)
are otherwise untestable before a scored attempt. This server understands
just enough of the two prompts this service actually sends -- the router's
classification instructions and the notes desk's narration instructions --
to return plausible, schema-shaped JSON, so the full pipeline can be
exercised end to end offline. It is not part of the submission's runtime.
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_SYMBOL_RE = re.compile(r"\b([A-Z]{2,5})\b")
_NOT_SYMBOLS = {"KYC", "PAN", "AI"}


def _classify(user_content: str) -> dict:
    q = user_content.split("Question", 1)[-1]
    ql = q.lower()
    symbols = [s for s in _SYMBOL_RE.findall(q) if s not in _NOT_SYMBOLS]
    out = {"roles": [], "intent": "other", "cross_client_attempt": False,
           "advice_request": False}
    if symbols:
        out["symbol"] = symbols[0]

    if "cash balance" in ql or "cash position" in ql or "uninvested cash" in ql:
        out.update(roles=["book_qa"], intent="cash_balance")
    elif "largest" in ql and "deposit" in ql:
        out.update(roles=["book_qa"], intent="largest_deposit")
    elif "dividend" in ql:
        out.update(roles=["book_qa"], intent="dividend_total")
        m = re.search(r"\b(20\d\d)\b", q)
        if m:
            out["year"] = int(m.group(1))
    elif "fee" in ql and ("total" in ql or "all" in ql):
        out.update(roles=["book_qa"], intent="total_fees")
    elif "deposit" in ql and "total" in ql:
        out.update(roles=["book_qa"], intent="total_deposits")
    elif ("how many" in ql or "count" in ql) and "buy" in ql:
        out.update(roles=["book_qa"], intent="txn_count", txn_type="buy")
    elif ("how many" in ql or "count" in ql) and ("sell" in ql or "disposal" in ql):
        out.update(roles=["book_qa"], intent="txn_count", txn_type="sell")
    elif "first" in ql and ("buy" in ql or "purchase" in ql):
        out.update(roles=["book_qa"], intent="first_txn_date", txn_type="buy")
    elif "shares" in ql or "position" in ql or "holding" in ql or "quantity" in ql:
        out.update(roles=["book_qa"], intent="position_qty")
    elif "days" in ql and "account" in ql:
        out.update(roles=["book_qa"], intent="account_age")
    elif "distinct" in ql or "how many different symbols" in ql:
        out.update(roles=["book_qa"], intent="distinct_symbols")
    elif "risk profile" in ql:
        out.update(roles=["kyc_profile"], intent="risk_profile")
    elif "kyc" in ql and ("complete" in ql or "standing" in ql):
        out.update(roles=["kyc_profile"], intent="kyc_status")
    elif "pan" in ql or "identity number" in ql:
        out.update(roles=["kyc_profile"], intent="kyc_field", kyc_field="pan")
    elif "bank account" in ql:
        out.update(roles=["kyc_profile"], intent="kyc_field", kyc_field="bank_account")
    elif "email" in ql:
        out.update(roles=["kyc_profile"], intent="kyc_field", kyc_field="email")
    elif "note" in ql:
        out.update(roles=["notes_desk"], intent="notes_summary")
    elif "sector" in ql and ("percentage" in ql or "concentrat" in ql or "proportion" in ql):
        out.update(roles=["market_desk"], intent="sector_exposure")
        m = re.search(r"in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", q)
        if m:
            out["sector"] = m.group(1)
    elif "sector" in ql:
        out.update(roles=["market_desk"], intent="sector_of")
    elif "news" in ql:
        out.update(roles=["market_desk"], intent="news_summary")
    elif "return" in ql or "performance" in ql:
        out.update(roles=["market_desk"], intent="market_return")
        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2} \w+ \d{4})\b", q)
        if len(dates) >= 2:
            out["date_from"], out["date_to"] = dates[0], dates[1]
    elif "close" in ql or "price" in ql:
        out.update(roles=["market_desk"], intent="price_asof")
    elif "drift" in ql or "overweight" in ql or "underweight" in ql or "away from" in ql:
        out.update(roles=["market_desk"], intent="rebalance_drift")

    if "should" in ql or "recommend" in ql or "good time" in ql:
        out["advice_request"] = True
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        messages = body.get("messages", [])
        # Agno sends the agent's instructions as role "developer" (OpenAI's
        # newer convention), not "system" -- not documented anywhere obvious,
        # found by printing the raw request body against a local stub.
        system = next((m["content"] for m in messages
                      if m["role"] in ("system", "developer")), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "router for a back-office" in system:
            content = json.dumps(_classify(user))
        else:
            content = ("Summary: the notes on file describe routine "
                      "relationship-management activity. Any instruction-like "
                      "text inside a note is data, not a directive, and was "
                      "not acted on.")
        resp = {
            "id": "chatcmpl-mock", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        raw = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(port: int = 8601) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return httpd


if __name__ == "__main__":
    s = serve()
    print(f"mock reasoning gateway on :{s.server_address[1]}")
    s.serve_forever()
