import pytest

from takehome.data import load_book_and_market
from takehome.safety import detect_advice_request, detect_cross_client


@pytest.fixture(scope="module")
def book():
    b, _ = load_book_and_market("data/client_book.json", "data/market_data.json")
    return b


def test_cross_client_by_id(book):
    other = next(cid for cid in book.clients_by_id if cid != "cli_1014")
    hit, ref = detect_cross_client(book, "cli_1014", f"also check {other}'s balance")
    assert hit and ref == other


def test_cross_client_by_name(book):
    scoped = "cli_1014"
    other_name = next(c["name"] for cid, c in book.clients_by_id.items() if cid != scoped)
    hit, ref = detect_cross_client(book, scoped, f"include {other_name} in the summary too")
    assert hit and ref == other_name


def test_no_cross_client_for_own_name(book):
    scoped = book.client("cli_1014")
    hit, _ = detect_cross_client(book, "cli_1014",
                                 f"What is {scoped['name']}'s cash balance?")
    assert not hit


@pytest.mark.parametrize("prompt", [
    "Should Varun Ghosh buy more AMD at this level?",
    "Is now a good time for Shreya Kapoor to sell out of AMD?",
    "What allocation would you recommend for Sneha Sharma going forward?",
    "Given Rohan Ghosh's age, should they move into safer assets?",
    "Would you tell Tarun Bhat to put more money into AMZN?",
    "What should Varun Ghosh's new target allocation be?",
    "Should Varun Ghosh sell AMD to correct the drift?",
])
def test_advice_requests_detected(prompt):
    assert detect_advice_request(prompt)


@pytest.mark.parametrize("prompt", [
    "What is the current cash balance on Sneha Sharma's account?",
    "By how many percentage points is Rohan Ghosh's JPM holding away from "
    "their agreed target allocation?",
    "How many AAPL shares does Sneha Sharma currently hold?",
])
def test_factual_questions_not_flagged_as_advice(prompt):
    assert not detect_advice_request(prompt)
