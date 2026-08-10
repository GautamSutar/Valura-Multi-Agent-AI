"""Maps a router Classification onto the deterministic retrieval layer, and
formats the result back into the pieces the response contract needs. This is
the only place intent names are wired to `retrieval.py` functions, so a new
question shape only ever means a new branch here.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

from takehome import retrieval as R
from takehome.data import Book, Market
from takehome.money import money2, pct2, qty_str

KYC_SYNONYMS = {
    "identity number": "pan", "identity_number": "pan", "pan number": "pan",
    "pan_number": "pan", "bank account number": "bank_account",
    "bank_account_number": "bank_account", "account number": "bank_account",
    "account_number": "bank_account", "bank account": "bank_account",
    "email address": "email", "phone": "mobile", "mobile number": "mobile",
    "dob": "date_of_birth", "income": "annual_income_band",
    "income band": "annual_income_band",
}

MONEY_INTENTS = {"cash_balance", "largest_deposit", "dividend_total",
                 "total_deposits", "total_fees"}
PCT_INTENTS = {"sector_exposure", "market_return", "rebalance_drift"}
COUNT_INTENTS = {"txn_count", "distinct_symbols", "news_summary"}
QTY_INTENTS = {"position_qty"}

INTENT_ROLE = {
    "cash_balance": "book_qa", "largest_deposit": "book_qa",
    "dividend_total": "book_qa", "total_deposits": "book_qa",
    "total_fees": "book_qa", "txn_count": "book_qa",
    "first_txn_date": "book_qa", "position_qty": "book_qa",
    "account_age": "book_qa", "distinct_symbols": "book_qa",
    "kyc_field": "kyc_profile", "risk_profile": "kyc_profile",
    "kyc_status": "kyc_profile",
    "notes_summary": "notes_desk", "txn_memo": "notes_desk",
    "price_asof": "market_desk", "market_return": "market_desk",
    "sector_of": "market_desk", "sector_exposure": "market_desk",
    "news_summary": "market_desk", "rebalance_drift": "market_desk",
}


@dataclass
class Fact:
    role: str
    intent: str
    result: R.QueryResult
    label: str = ""          # a short human phrase for prose, e.g. "PAN"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s.strip()[:10])
    except ValueError:
        return None


def _month_range(month: str | None) -> tuple[str | None, str | None]:
    if not month:
        return None, None
    try:
        y, m = (int(x) for x in month.split("-")[:2])
        last = calendar.monthrange(y, m)[1]
        return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"
    except (ValueError, IndexError):
        return None, None


def _normalize_kyc_field(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    return KYC_SYNONYMS.get(key, key.replace(" ", "_"))


def resolve(book: Book, market: Market, client: dict, intent: str,
           cls) -> Fact | None:
    """Runs one intent against the deterministic layer. Returns None for an
    intent this dispatcher does not recognise (including 'other'), which the
    orchestrator treats as an honest abstention."""
    date_from, date_to = cls.date_from, cls.date_to
    if cls.month and not (date_from or date_to):
        date_from, date_to = _month_range(cls.month)

    if intent == "cash_balance":
        return Fact("book_qa", intent, R.cash_balance(client, _parse_date(cls.asof)),
                   "cash balance")
    if intent == "largest_deposit":
        return Fact("book_qa", intent, R.largest_deposit(client), "largest deposit")
    if intent == "dividend_total":
        if not cls.symbol:
            return Fact("book_qa", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "dividend total")
        return Fact("book_qa", intent,
                   R.dividend_total(client, cls.symbol, cls.year), "dividend total")
    if intent == "total_deposits":
        return Fact("book_qa", intent,
                   R.total_deposits(client, date_from, date_to, cls.year),
                   "total deposits")
    if intent == "total_fees":
        return Fact("book_qa", intent, R.total_fees(client), "total fees")
    if intent == "txn_count":
        ttype = cls.txn_type or "buy"
        return Fact("book_qa", intent,
                   R.count_transactions(client, ttype, cls.symbol, date_from, date_to),
                   f"{ttype} count")
    if intent == "first_txn_date":
        if not cls.symbol:
            return Fact("book_qa", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "first transaction date")
        ttype = cls.txn_type or "buy"
        return Fact("book_qa", intent,
                   R.first_transaction_date(client, ttype, cls.symbol),
                   "first transaction date")
    if intent == "position_qty":
        if not cls.symbol:
            return Fact("book_qa", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "position size")
        asof = _parse_date(cls.asof)
        result = (R.position_qty_asof(client, cls.symbol, asof) if asof
                 else R.position_qty_current(book, client, cls.symbol))
        return Fact("book_qa", intent, result, "position size")
    if intent == "account_age":
        return Fact("book_qa", intent, R.account_age_days(book, client), "account age")
    if intent == "distinct_symbols":
        asof = _parse_date(cls.asof) or book.as_of
        return Fact("book_qa", intent, R.distinct_symbols_asof(client, asof),
                   "distinct holdings")

    if intent == "kyc_field":
        field_name = _normalize_kyc_field(cls.kyc_field)
        result = R.kyc_field(client, field_name or "")
        return Fact("kyc_profile", intent, result, field_name or "field")
    if intent == "risk_profile":
        return Fact("kyc_profile", intent, R.risk_profile(client), "risk profile")
    if intent == "kyc_status":
        return Fact("kyc_profile", intent, R.kyc_status(client), "KYC status")

    if intent == "notes_summary":
        notes = R.notes_for(client)
        cites = [n["id"] for n in notes]
        result = R.QueryResult(len(notes), cites, found=bool(notes),
                               note="no notes on file" if not notes else "")
        return Fact("notes_desk", intent, result, "notes")
    if intent == "txn_memo":
        if not cls.txn_id:
            return Fact("notes_desk", intent, R.QueryResult(None, [], found=False,
                        note="no transaction id was named"), "transaction memo")
        return Fact("notes_desk", intent, R.memo_for_transaction(client, cls.txn_id),
                   "transaction memo")

    if intent == "price_asof":
        if not cls.symbol:
            return Fact("market_desk", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "price")
        asof = _parse_date(cls.asof) or market.meta and _parse_date(market.meta.get("as_of"))
        return Fact("market_desk", intent, R.price_asof(market, cls.symbol, asof),
                   "price")
    if intent == "market_return":
        df, dt = _parse_date(date_from), _parse_date(date_to)
        if not (cls.symbol and df and dt):
            return Fact("market_desk", intent, R.QueryResult(None, [], found=False,
                        note="the instrument or the date range was not fully specified"),
                       "return")
        return Fact("market_desk", intent, R.market_return(market, cls.symbol, df, dt),
                   "return")
    if intent == "sector_of":
        if not cls.symbol:
            return Fact("market_desk", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "sector")
        return Fact("market_desk", intent, R.sector_of(market, cls.symbol), "sector")
    if intent == "sector_exposure":
        if not cls.sector:
            return Fact("market_desk", intent, R.QueryResult(None, [], found=False,
                        note="no sector was named"), "sector exposure")
        return Fact("market_desk", intent,
                   R.sector_exposure(book, market, client, cls.sector),
                   "sector exposure")
    if intent == "news_summary":
        if not cls.symbol:
            return Fact("market_desk", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "news coverage")
        return Fact("market_desk", intent,
                   R.news_for_symbol(market, cls.symbol, _parse_date(cls.asof)),
                   "news coverage")
    if intent == "rebalance_drift":
        if not cls.symbol:
            return Fact("market_desk", intent, R.QueryResult(None, [], found=False,
                        note="no instrument was named"), "allocation drift")
        return Fact("market_desk", intent,
                   R.rebalance_drift(book, market, client, cls.symbol),
                   "allocation drift")
    return None


def format_value(intent: str, result: R.QueryResult) -> str | None:
    if result.value is None:
        return None
    if intent in MONEY_INTENTS:
        return money2(result.value)
    if intent in PCT_INTENTS:
        return pct2(result.value)
    if intent in QTY_INTENTS:
        return qty_str(result.value)
    if intent in COUNT_INTENTS:
        return str(int(result.value))
    if intent == "account_age":
        return str(int(result.value))
    if intent == "kyc_field":
        return str(result.value)
    return str(result.value)


def masked_kyc_value(field_name: str, raw_value) -> str:
    if field_name == "pan":
        return mask_pan(str(raw_value))
    if field_name == "bank_account":
        acct = raw_value.get("account_number") if isinstance(raw_value, dict) else raw_value
        return mask_bank_account(str(acct))
    return str(raw_value)
