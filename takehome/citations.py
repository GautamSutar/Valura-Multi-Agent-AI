"""The citation rule from the brief: cite the records relied on; if more than
six, cite the client id instead of listing them all."""
from __future__ import annotations

MAX_INDIVIDUAL_CITATIONS = 6


def build_citations(client_id: str, record_ids: list[str]) -> list[str]:
    ids = list(dict.fromkeys(record_ids))  # de-dupe, keep order
    if len(ids) > MAX_INDIVIDUAL_CITATIONS:
        return [client_id]
    return ids
