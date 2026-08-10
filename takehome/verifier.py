"""The verifier: a deterministic gate that runs on every answer right before
it leaves the service. It never calls a model. Its job is to catch the two
failures the brief treats as submission-ending -- a citation or a stray
string that discloses another client's data, and an unmasked identifier --
regardless of which code path produced the answer.
"""
from __future__ import annotations

import re

from takehome.data import Book
from takehome.masking import mask_bank_account, mask_pan

_MASK_MARKER = "[masked]"


def owned_citations(book: Book, scoped_client_id: str, citations: list[str]) -> list[str]:
    """Citations that resolve to a different client. These must never leave
    the service; a citation is disclosure, not an audit trail, once it names
    someone else's record."""
    bad = []
    for c in citations:
        owner = book.owner_of.get(c)
        if owner is not None and owner != scoped_client_id and c != scoped_client_id:
            bad.append(c)
    return bad


def scrub_own_pii(text: str, client: dict) -> str:
    """Belt-and-braces: even though raw PAN/bank values never enter a
    prompt, strip them from any free text one more time before it leaves,
    so no code path can accidentally bypass the mask."""
    if not text:
        return text
    kyc = client.get("kyc", {})
    pan = kyc.get("pan")
    bank = (kyc.get("bank_account") or {}).get("account_number")
    for raw in (pan, bank):
        if raw and raw in text:
            masked = mask_pan(raw) if raw == pan else mask_bank_account(raw)
            text = text.replace(raw, masked)
    return text


def scan_cross_client_leak(book: Book, scoped_client_id: str, prompt: str,
                           text: str) -> list[str]:
    """Independent scan of the final answer text for another client's name
    or PII, run regardless of how the answer was produced -- this is the
    last line of defence if an LLM-generated summary drifted."""
    hits = []
    scoped = book.client(scoped_client_id) or {}
    for cid, client in book.clients_by_id.items():
        if cid == scoped_client_id:
            continue
        name = client.get("name", "")
        if name and name in text and name.lower() not in prompt.lower():
            hits.append(f"names another client ({cid})")
        kyc = client.get("kyc", {})
        for raw in (kyc.get("pan"), (kyc.get("bank_account") or {}).get("account_number")):
            if raw and raw in text:
                hits.append(f"exposes an identity/bank value belonging to {cid}")
    return hits
