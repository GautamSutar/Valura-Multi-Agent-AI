"""Regex backstops for the two failures the brief treats as submission-ending
rather than a deduction: cross-client disclosure and personalised advice.

These run unconditionally, independently of what the router agent decides,
and the orchestrator ORs them with the router's own judgement. A classifier
that is merely usually right is not good enough for a check this expensive
to get wrong.
"""
from __future__ import annotations

import re

from takehome.data import Book

_CLIENT_ID_RE = re.compile(r"\bcli_\d+\b")

_ADVICE_PATTERNS = [
    r"\bshould\b[^.?!]{0,80}\b(buy|sell|invest|switch|move|increase|reduce|exit|hold|add|trim)\b",
    r"\bis (now|this) a good time\b",
    r"\bwould you recommend\b",
    r"\ballocation would you recommend\b",
    r"\bwhat should\b[^.?!]{0,60}\ballocation\b",
    r"\bwould you (tell|advise|suggest)\b",
    r"\bshould (they|he|she|the client)\b",
    r"\bgood idea to (buy|sell|add|trim|exit)\b",
    r"\bwhat allocation would suit\b",
]
_ADVICE_RE = [re.compile(p, re.I) for p in _ADVICE_PATTERNS]


def detect_advice_request(prompt: str) -> bool:
    return any(rx.search(prompt) for rx in _ADVICE_RE)


def detect_cross_client(book: Book, scoped_client_id: str, prompt: str) -> tuple[bool, str | None]:
    """Returns (attempted, other_client_name_or_id). Catches an explicit
    cli_NNNN reference to a different client and any other client's full
    name appearing in the prompt -- however the request is framed."""
    for m in _CLIENT_ID_RE.findall(prompt):
        if m != scoped_client_id and m in book.clients_by_id:
            return True, m
    lowered = prompt.lower()
    scoped_name = (book.client(scoped_client_id) or {}).get("name", "")
    for cid, client in book.clients_by_id.items():
        if cid == scoped_client_id:
            continue
        name = client.get("name", "")
        if name and name != scoped_name and name.lower() in lowered:
            return True, name
    return False, None
