"""Book and market loading, held in memory once at startup.

Nothing here re-reads the file per question, and nothing here puts whole
records into an LLM prompt: callers get indexed, client-scoped access so
retrieval stays cheap and citations stay auditable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class Book:
    meta: dict
    clients_by_id: dict[str, dict]
    # record id -> owning client id, built once, for cross-client guards.
    owner_of: dict[str, str] = field(default_factory=dict)

    @property
    def as_of(self) -> date:
        return date.fromisoformat(self.meta["as_of"])

    def client(self, client_id: str) -> dict | None:
        return self.clients_by_id.get(client_id)

    @classmethod
    def load(cls, path: str) -> "Book":
        raw = _load_json(path)
        clients_by_id = {c["id"]: c for c in raw["clients"]}
        owner: dict[str, str] = {}
        for c in raw["clients"]:
            cid = c["id"]
            owner[cid] = cid
            owner[c["kyc"]["id"]] = cid
            for a in c.get("accounts", []):
                owner[a["id"]] = cid
            for r in c.get("suitability_reviews", []):
                owner[r["id"]] = cid
            for n in c.get("notes", []):
                owner[n["id"]] = cid
            for t in c.get("transactions", []):
                owner[t["id"]] = cid
            for p in c.get("positions_snapshot", []):
                owner[p["id"]] = cid
        return cls(meta=raw["meta"], clients_by_id=clients_by_id, owner_of=owner)


@dataclass
class Market:
    meta: dict
    instruments_by_symbol: dict[str, dict]
    prices_by_symbol: dict[str, list[dict]]
    news: list[dict]

    @property
    def covered_symbols(self) -> set[str]:
        return set(self.meta.get("covered_symbols", []))

    def is_covered(self, symbol: str) -> bool:
        return symbol in self.covered_symbols

    def sector_of(self, symbol: str) -> str | None:
        inst = self.instruments_by_symbol.get(symbol)
        return inst.get("sector") if inst else None

    @classmethod
    def load(cls, path: str) -> "Market":
        raw = _load_json(path)
        instruments = {i["symbol"]: i for i in raw.get("instruments", [])}
        prices = raw.get("prices", {})
        return cls(meta=raw.get("meta", {}), instruments_by_symbol=instruments,
                   prices_by_symbol=prices, news=raw.get("news", []))


def load_book_and_market(book_path: str, market_path: str) -> tuple[Book, Market]:
    return Book.load(book_path), Market.load(market_path)
