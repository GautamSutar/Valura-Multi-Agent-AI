import pytest

from takehome.dispatch import normalize_txn_type


@pytest.mark.parametrize("raw, expected", [
    ("buy", "buy"), ("buys", "buy"), ("purchase", "buy"), ("purchases", "buy"),
    ("sell", "sell"), ("sells", "sell"), ("sale", "sell"), ("disposal", "sell"),
    ("deposit", "deposit"), ("deposits", "deposit"), ("funding", "deposit"),
    ("withdrawal", "withdrawal"), ("withdraw", "withdrawal"),
    ("dividend", "dividend"), ("dividends", "dividend"),
    ("fee", "fee"), ("fees", "fee"),
    (None, None), ("", None), ("gibberish", None),
])
def test_normalize_txn_type(raw, expected):
    assert normalize_txn_type(raw) == expected
