"""Deterministic retrieval and arithmetic over a single client's records.

Every function here is scoped to one client dict (the caller is responsible
for having looked it up by the question's client_id) and returns a
`QueryResult`: a value, the record ids it rests on, and whether the two
sources that could answer it disagree. No LLM touches any of this; the
agents only narrate what these functions already computed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from takehome.data import Book, Market
from takehome.masking import mask_bank_account, mask_pan
from takehome.money import dec

CASH_QTY_TOLERANCE = Decimal("0.0005")

# Fields genuinely present on a KYC record. Anything asked about outside this
# set (email, mobile, employer, nominee, ...) does not exist in this book at
# all, so the only correct move is to abstain rather than guess.
KYC_AVAILABLE_FIELDS = {
    "kyc_status", "risk_profile", "date_of_birth", "address",
    "annual_income_band", "pan", "bank_account",
}


@dataclass
class QueryResult:
    value: object                    # Decimal | str | int | None
    citations: list[str] = field(default_factory=list)
    conflict: bool = False
    conflict_citations: list[str] = field(default_factory=list)
    found: bool = True                # False => nothing in the book answers this
    note: str = ""


def _txn_date(t: dict) -> str:
    return t["date"]


def _cash_delta(t: dict) -> Decimal:
    ty = t["type"]
    if ty == "deposit":
        return dec(t["amount_usd"])
    if ty == "withdrawal":
        return -dec(t["amount_usd"])
    if ty == "buy":
        return -dec(t["net_usd"])
    if ty == "sell":
        return dec(t["net_usd"])
    if ty == "dividend":
        return dec(t["net_usd"])
    if ty == "fee":
        return -dec(t["amount_usd"])
    return Decimal("0")


# -- book_qa ------------------------------------------------------------

def cash_balance(client: dict, asof: date | None = None) -> QueryResult:
    txns = client["transactions"]
    if asof is not None:
        txns = [t for t in txns if t["date"] <= asof.isoformat()]
    bal = sum((_cash_delta(t) for t in txns), Decimal("0"))
    return QueryResult(bal, [client["id"]])


def largest_deposit(client: dict) -> QueryResult:
    deps = [t for t in client["transactions"] if t["type"] == "deposit"]
    if not deps:
        return QueryResult(None, [], found=False, note="no deposits on record")
    best = max(deps, key=lambda t: dec(t["amount_usd"]))
    return QueryResult(dec(best["amount_usd"]), [best["id"]])


def dividend_total(client: dict, symbol: str, year: int | None = None) -> QueryResult:
    divs = [t for t in client["transactions"]
            if t["type"] == "dividend" and t.get("symbol") == symbol]
    if year is not None:
        divs = [t for t in divs if t["date"].startswith(str(year))]
    if not divs:
        return QueryResult(None, [], found=False,
                           note=f"no {symbol} dividends on record for the period asked")
    total = sum((dec(t["net_usd"]) for t in divs), Decimal("0"))
    return QueryResult(total, [t["id"] for t in divs])


def total_fees(client: dict) -> QueryResult:
    fees = [t for t in client["transactions"] if t["type"] == "fee"]
    total = sum((dec(t["amount_usd"]) for t in fees), Decimal("0"))
    return QueryResult(total, [client["id"]])


def total_deposits(client: dict, date_from: str | None = None,
                   date_to: str | None = None, year: int | None = None) -> QueryResult:
    deps = [t for t in client["transactions"] if t["type"] == "deposit"]
    if year is not None:
        deps = [t for t in deps if t["date"].startswith(str(year))]
    if date_from is not None:
        deps = [t for t in deps if t["date"] >= date_from]
    if date_to is not None:
        deps = [t for t in deps if t["date"] <= date_to]
    total = sum((dec(t["amount_usd"]) for t in deps), Decimal("0"))
    return QueryResult(total, [client["id"]])


def count_transactions(client: dict, txn_type: str, symbol: str | None = None,
                       date_from: str | None = None, date_to: str | None = None
                       ) -> QueryResult:
    txns = [t for t in client["transactions"] if t["type"] == txn_type]
    if symbol is not None:
        txns = [t for t in txns if t.get("symbol") == symbol]
    if date_from is not None:
        txns = [t for t in txns if t["date"] >= date_from]
    if date_to is not None:
        txns = [t for t in txns if t["date"] <= date_to]
    return QueryResult(len(txns), [t["id"] for t in txns] or [client["id"]])


def first_transaction_date(client: dict, txn_type: str, symbol: str) -> QueryResult:
    txns = sorted((t for t in client["transactions"]
                   if t["type"] == txn_type and t.get("symbol") == symbol),
                  key=_txn_date)
    if not txns:
        return QueryResult(None, [], found=False,
                           note=f"no {txn_type} of {symbol} on record")
    return QueryResult(txns[0]["date"], [txns[0]["id"]])


def _replayed_qty(client: dict, symbol: str, asof: date | None) -> Decimal:
    qty = Decimal("0")
    for t in client["transactions"]:
        if t.get("symbol") != symbol:
            continue
        if asof is not None and t["date"] > asof.isoformat():
            continue
        if t["type"] == "buy":
            qty += dec(t["quantity"])
        elif t["type"] == "sell":
            qty -= dec(t["quantity"])
    return qty


def position_qty_asof(client: dict, symbol: str, asof: date) -> QueryResult:
    qty = _replayed_qty(client, symbol, asof)
    return QueryResult(qty, [client["id"]])


def position_qty_current(book: Book, client: dict, symbol: str) -> QueryResult:
    """'Current' holdings are authoritative from the positions snapshot, but
    cross-checked against the full transaction replay. Two records in this
    book can genuinely disagree; when they do, that is a conflict to surface,
    not a value to pick."""
    snap = next((p for p in client["positions_snapshot"] if p["symbol"] == symbol), None)
    replayed = _replayed_qty(client, symbol, book.as_of)
    txn_ids = [t["id"] for t in client["transactions"] if t.get("symbol") == symbol
               and t["type"] in ("buy", "sell")]
    if snap is None:
        if replayed == 0:
            return QueryResult(None, [], found=False,
                               note=f"no {symbol} position on record")
        # No snapshot line, but transactions imply a position: still a fact.
        return QueryResult(replayed, [client["id"]])
    snap_qty = dec(snap["quantity"])
    if abs(snap_qty - replayed) > CASH_QTY_TOLERANCE:
        return QueryResult(
            None, [snap["id"]], conflict=True,
            conflict_citations=[snap["id"], *txn_ids],
            note=("the positions snapshot and the transaction history do not "
                  "agree on this quantity"))
    return QueryResult(snap_qty, [snap["id"]])


def distinct_symbols_asof(client: dict, asof: date) -> QueryResult:
    symbols = {t["symbol"] for t in client["transactions"] if t.get("symbol")}
    held = []
    for s in symbols:
        if _replayed_qty(client, s, asof) > CASH_QTY_TOLERANCE:
            held.append(s)
    return QueryResult(len(held), [client["id"]])


def account_age_days(book: Book, client: dict) -> QueryResult:
    acc = client["accounts"][0] if client.get("accounts") else None
    if acc is None:
        return QueryResult(None, [], found=False, note="no account record on file")
    opened = date.fromisoformat(acc["opened"])
    days = (book.as_of - opened).days
    return QueryResult(days, [acc["id"]])


# -- kyc_profile ----------------------------------------------------------

def kyc_field(client: dict, field_name: str) -> QueryResult:
    if field_name not in KYC_AVAILABLE_FIELDS:
        return QueryResult(None, [], found=False,
                           note=f"{field_name} is not a field this book records for KYC")
    kyc = client["kyc"]
    val = kyc.get(field_name)
    if val is None:
        return QueryResult(None, [], found=False,
                           note=f"no {field_name} on file")
    # Masking happens here, at the source, so the raw PAN or account number
    # never exists in a QueryResult, a prompt, or a response: nothing
    # downstream can bypass a mask that was never carried past this line.
    if field_name == "pan":
        val = mask_pan(str(val))
    elif field_name == "bank_account":
        val = mask_bank_account(str(val.get("account_number", "")))
    return QueryResult(val, [kyc["id"]])


def risk_profile(client: dict) -> QueryResult:
    kyc = client["kyc"]
    kyc_risk = kyc.get("risk_profile")
    reviews = client.get("suitability_reviews") or []
    if not reviews:
        return QueryResult(kyc_risk, [kyc["id"]])
    latest = max(reviews, key=lambda r: r["date"])
    if latest.get("risk_profile") and latest["risk_profile"] != kyc_risk:
        return QueryResult(
            None, [kyc["id"]], conflict=True,
            conflict_citations=[kyc["id"], latest["id"]],
            note="the KYC record and the latest suitability review disagree "
                 "on risk profile")
    return QueryResult(kyc_risk, [kyc["id"]])


def kyc_status(client: dict) -> QueryResult:
    kyc = client["kyc"]
    return QueryResult(kyc.get("kyc_status"), [kyc["id"]])


# -- notes_desk -------------------------------------------------------------

def notes_for(client: dict) -> list[dict]:
    return list(client.get("notes") or [])


def memo_for_transaction(client: dict, txn_id: str) -> QueryResult:
    for t in client["transactions"]:
        if t["id"] == txn_id and t.get("description"):
            return QueryResult(t["description"], [t["id"]])
    return QueryResult(None, [], found=False,
                       note=f"no memo on record for {txn_id}")


# -- market_desk --------------------------------------------------------

def price_asof(market: Market, symbol: str, asof: date) -> QueryResult:
    if not market.is_covered(symbol):
        return QueryResult(None, [], found=False,
                           note=f"{symbol} is not in the covered market dataset")
    series = market.prices_by_symbol.get(symbol, [])
    candidates = [p for p in series if p["date"] <= asof.isoformat()]
    if not candidates:
        return QueryResult(None, [], found=False,
                           note=f"no {symbol} close on or before {asof.isoformat()}")
    best = max(candidates, key=lambda p: p["date"])
    return QueryResult(dec(best["close"]), [symbol], note=f"as at {best['date']}")


def market_return(market: Market, symbol: str, date_from: date, date_to: date) -> QueryResult:
    if not market.is_covered(symbol):
        return QueryResult(None, [], found=False,
                           note=f"{symbol} is not in the covered market dataset")
    p0 = price_asof(market, symbol, date_from)
    p1 = price_asof(market, symbol, date_to)
    if not p0.found or not p1.found or p0.value in (None, Decimal("0")):
        return QueryResult(None, [], found=False,
                           note="insufficient price coverage over that period")
    ret = (p1.value - p0.value) / p0.value * Decimal("100")
    return QueryResult(ret, [symbol])


def sector_of(market: Market, symbol: str) -> QueryResult:
    if not market.is_covered(symbol):
        return QueryResult(None, [], found=False,
                           note=f"{symbol} is not in the covered market dataset")
    inst = market.instruments_by_symbol[symbol]
    return QueryResult(inst["sector"], [symbol])


def sector_exposure(book: Book, market: Market, client: dict, sector: str) -> QueryResult:
    positions = client.get("positions_snapshot") or []
    total = sum((dec(p["market_value_usd"]) for p in positions), Decimal("0"))
    if total == 0:
        return QueryResult(None, [], found=False, note="no positions on record")
    matched = [p for p in positions if market.sector_of(p["symbol"]) == sector]
    sector_val = sum((dec(p["market_value_usd"]) for p in matched), Decimal("0"))
    pct = sector_val / total * Decimal("100")
    cites = [p["id"] for p in matched] or [client["id"]]
    return QueryResult(pct, cites)


def news_for_symbol(market: Market, symbol: str, asof: date | None = None) -> QueryResult:
    if not market.is_covered(symbol):
        return QueryResult(None, [], found=False,
                           note=f"{symbol} is not in the covered market dataset")
    items = [n for n in market.news if n["symbol"] == symbol]
    if asof is not None:
        items = [n for n in items if n["date"] <= asof.isoformat()]
    return QueryResult(len(items), [n["id"] for n in items], note="news items")


def rebalance_drift(book: Book, market: Market, client: dict, symbol: str) -> QueryResult:
    positions = client.get("positions_snapshot") or []
    total = sum((dec(p["market_value_usd"]) for p in positions), Decimal("0"))
    reviews = client.get("suitability_reviews") or []
    if not reviews:
        return QueryResult(None, [], found=False, note="no target allocation on file")
    latest = max(reviews, key=lambda r: r["date"])
    target = latest.get("target_allocation_pct", {}).get(symbol)
    if target is None:
        return QueryResult(None, [], found=False,
                           note=f"no agreed target allocation for {symbol} on file")
    pos = next((p for p in positions if p["symbol"] == symbol), None)
    current_val = dec(pos["market_value_usd"]) if pos else Decimal("0")
    current_pct = (current_val / total * Decimal("100")) if total else Decimal("0")
    drift = current_pct - dec(target)
    cites = [latest["id"]] + ([pos["id"]] if pos else [])
    return QueryResult(drift, cites)
