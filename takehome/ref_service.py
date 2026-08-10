"""Adapter for `harness/reference_client.py`, which drives the real
assessment protocol and expects `from takehome.ref_service import Reference`
with the constructor `Reference(book, llm_base_url, api_key=..., market=...)`
and an object exposing `.roster()` and `.answer(question_id, client_id,
prompt)`. Answerer already implements exactly that (and accepts the book/
market either as the raw dicts the live server returns from GET /v1/book and
GET /v1/market, or as already-loaded Book/Market objects), so this module is
just the import path the harness looks for.
"""
from __future__ import annotations

from takehome.orchestrator import Answerer as Reference

__all__ = ["Reference"]
