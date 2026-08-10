"""Pins the deterministic retrieval/arithmetic layer against real values from
the practice key, so a regression here is caught before it ever reaches the
scorer."""
from datetime import date
from decimal import Decimal

import pytest

from takehome.data import load_book_and_market
from takehome import retrieval as R

BOOK_PATH = "data/client_book.json"
MARKET_PATH = "data/market_data.json"


@pytest.fixture(scope="module")
def bm():
    return load_book_and_market(BOOK_PATH, MARKET_PATH)


def test_cash_balance(bm):
    book, _ = bm
    c = book.client("cli_1014")
    r = R.cash_balance(c)
    assert r.value == Decimal("15386.78")
    assert r.citations == ["cli_1014"]


def test_cash_balance_asof(bm):
    book, _ = bm
    c = book.client("cli_1014")
    r = R.cash_balance(c, asof=date(2026, 7, 28))
    assert r.value == Decimal("55112.64")


def test_largest_deposit(bm):
    book, _ = bm
    c = book.client("cli_1014")
    r = R.largest_deposit(c)
    assert r.value == Decimal("19342.61")
    assert r.citations == ["txn_104543"]


def test_dividend_total(bm):
    book, _ = bm
    c = book.client("cli_1024")
    r = R.dividend_total(c, "MSFT", year=2024)
    assert r.value == Decimal("7.13")
    assert set(r.citations) == {"txn_108015", "txn_108234"}


def test_position_qty_current(bm):
    book, _ = bm
    c = book.client("cli_1014")
    r = R.position_qty_current(book, c, "AAPL")
    assert r.value == Decimal("2.9849")
    assert r.citations == ["pos_1014_AAPL"]
    assert not r.conflict


def test_position_qty_conflict(bm):
    book, _ = bm
    c = book.client("cli_1022")
    r = R.position_qty_current(book, c, "AAPL")
    assert r.conflict
    assert r.value is None
    assert "pos_1022_AAPL" in r.conflict_citations


def test_position_qty_asof(bm):
    book, _ = bm
    c = book.client("cli_1014")
    r = R.position_qty_asof(c, "AAPL", date(2026, 7, 10))
    assert r.value == Decimal("7.9008")


def test_account_age_days(bm):
    book, _ = bm
    c = book.client("cli_1024")
    r = R.account_age_days(book, c)
    assert r.value == 747
    assert r.citations == ["acc_1024"]


def test_risk_profile_conflict(bm):
    book, _ = bm
    c = book.client("cli_1010")
    r = R.risk_profile(c)
    assert r.conflict


def test_kyc_unavailable_field_abstains(bm):
    book, _ = bm
    c = book.client("cli_1012")
    r = R.kyc_field(c, "email")
    assert not r.found


def test_sector_exposure(bm):
    book, market = bm
    c = book.client("cli_1021")
    r = R.sector_exposure(book, market, c, "Communication Services")
    assert r.value.quantize(Decimal("0.01")) == Decimal("27.18")


def test_market_return(bm):
    _, market = bm
    r = R.market_return(market, "AMD", date(2025, 7, 1), date(2026, 7, 1))
    assert r.value.quantize(Decimal("0.01")) == Decimal("-4.80")


def test_rebalance_drift(bm):
    book, market = bm
    c = book.client("cli_1006")
    r = R.rebalance_drift(book, market, c, "JPM")
    assert r.value.quantize(Decimal("0.01")) == Decimal("-32.15")


def test_uncovered_symbol_abstains(bm):
    _, market = bm
    r = R.price_asof(market, "PFE", date(2026, 7, 1))
    assert not r.found
    r2 = R.market_return(market, "PFE", date(2025, 7, 1), date(2026, 7, 1))
    assert not r2.found
    r3 = R.sector_of(market, "PFE")
    assert not r3.found
