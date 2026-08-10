"""The HTTP service: GET /health, GET /agents, POST /answer.

Reads BOOK_PATH, MARKET_PATH, LLM_BASE_URL, LLM_API_KEY and PORT from the
environment per the kit's contract, loads the book and market exactly once
at startup, and builds one Answerer shared across requests -- no per-request
reload, no per-question refetch.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from takehome.data import load_book_and_market
from takehome.orchestrator import Answerer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("takehome.service")

app = FastAPI(title="Valura take-home answer service")

_state: dict = {}


@app.on_event("startup")
def _startup() -> None:
    book_path = os.environ["BOOK_PATH"]
    market_path = os.environ["MARKET_PATH"]
    llm_base_url = os.environ["LLM_BASE_URL"]
    llm_api_key = os.environ["LLM_API_KEY"]
    book, market = load_book_and_market(book_path, market_path)
    _state["answerer"] = Answerer(book, llm_base_url, llm_api_key, market)
    logger.info("loaded %d clients, %d covered symbols",
               len(book.clients_by_id), len(market.covered_symbols))


@app.get("/health")
def health():
    return {"status": "ok" if "answerer" in _state else "starting"}


@app.get("/agents")
def agents():
    return _state["answerer"].roster()


@app.post("/answer")
async def answer(request: Request):
    body = await request.json()
    question_id = body.get("question_id", "")
    client_id = body.get("client_id", "")
    prompt = body.get("prompt", "")
    try:
        resp = _state["answerer"].answer(question_id, client_id, prompt)
    except Exception:  # noqa: BLE001 - a crash here forfeits every question
        logger.exception("unhandled error answering %s", question_id)
        resp = {"question_id": question_id, "answer": "", "answer_value": None,
               "abstained": True, "refused": False,
               "reason": "internal error while answering this question",
               "citations": [], "confidence": 0.0, "flags": [], "agents": ["router"]}
    return JSONResponse(resp)
