"""Pins the intent-normalization layer against the actual near-miss values a
real reasoning model returned during a qualifying attempt (e.g. 'balance'
instead of 'cash_balance'), which is what forced `Classification.intent` to
become a free string instead of a strict Literal -- see dispatch.py and
agents.py for why."""
import pytest

from takehome.dispatch import normalize_intent


@pytest.mark.parametrize("raw, expected", [
    ("cash_balance", "cash_balance"),
    ("balance", "cash_balance"),
    ("cash", "cash_balance"),
    ("Cash Balance", "cash_balance"),
    ("deposit", "total_deposits"),
    ("largest_deposit", "largest_deposit"),
    ("biggest_deposit", "largest_deposit"),
    ("dividend", "dividend_total"),
    ("fee", "total_fees"),
    ("buy_count", "txn_count"),
    ("first_buy", "first_txn_date"),
    ("holding", "position_qty"),
    ("shares", "position_qty"),
    ("age", "account_age"),
    ("symbols", "distinct_symbols"),
    ("pan", "kyc_field"),
    ("risk", "risk_profile"),
    ("kyc_complete", "kyc_status"),
    ("notes", "notes_summary"),
    ("memo", "txn_memo"),
    ("close", "price_asof"),
    ("return", "market_return"),
    ("sector", "sector_of"),
    ("sector_concentration", "sector_exposure"),
    ("news", "news_summary"),
    ("drift", "rebalance_drift"),
    ("other", "other"),
    (None, "other"),
    ("", "other"),
    ("completely_unrelated_gibberish", "other"),
])
def test_normalize_intent(raw, expected):
    assert normalize_intent(raw) == expected


def test_role_name_does_not_resolve_to_itself():
    # book_qa is a role, not an intent -- it should fall through to "other"
    # rather than being (mis)treated as a valid intent.
    assert normalize_intent("book_qa") == "other"
