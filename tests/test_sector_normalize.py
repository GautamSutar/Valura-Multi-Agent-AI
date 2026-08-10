from takehome.data import load_book_and_market
from takehome.dispatch import _normalize_sector


def test_sector_normalize_case_insensitive():
    _, market = load_book_and_market("data/client_book.json", "data/market_data.json")
    assert _normalize_sector(market, "communication services") == "Communication Services"
    assert _normalize_sector(market, "COMMUNICATION SERVICES") == "Communication Services"


def test_sector_normalize_substring():
    _, market = load_book_and_market("data/client_book.json", "data/market_data.json")
    assert _normalize_sector(market, "technology") == "Information Technology"


def test_sector_normalize_none():
    _, market = load_book_and_market("data/client_book.json", "data/market_data.json")
    assert _normalize_sector(market, None) is None
