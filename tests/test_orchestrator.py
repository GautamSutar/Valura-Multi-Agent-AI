"""End-to-end tests through the orchestrator, against the mock reasoning
gateway (see mock_reasoning_gateway.py) so the LLM-dependent paths --
classification, cross-client and advice refusal, notes narration -- are
exercised offline without a real provider key. The bundled stub gateway
cannot do this: it returns a fixed acknowledgement string regardless of
prompt, so it only rehearses plumbing, not reasoning.
"""
import threading

import pytest

from takehome.data import load_book_and_market
from takehome.orchestrator import Answerer
from takehome.schema_check import errors as schema_errors
from tests.mock_reasoning_gateway import serve


@pytest.fixture(scope="module")
def mock_gateway():
    srv = serve(port=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


@pytest.fixture(scope="module")
def answerer(mock_gateway):
    book, market = load_book_and_market("data/client_book.json", "data/market_data.json")
    return Answerer(book, mock_gateway, "x", market)


def _assert_valid(resp):
    errs = schema_errors(resp)
    assert not errs, errs
    return resp


def test_exact_value_grounded(answerer):
    r = _assert_valid(answerer.answer(
        "q_001", "cli_1014", "What is the current cash balance on Sneha Sharma's account?"))
    assert r["answer_value"] == "15386.78"
    assert r["citations"] == ["cli_1014"]
    assert not r["abstained"] and not r["refused"]
    assert "router" in r["agents"] and "book_qa" in r["agents"]


def test_pii_is_masked(answerer):
    r = _assert_valid(answerer.answer(
        "q_037", "cli_1014", "What is the PAN on file for Sneha Sharma?"))
    assert r["answer_value"] == "****249H"
    assert "BSNZA2249H" not in r["answer"]


def test_cross_client_refused(answerer):
    r = _assert_valid(answerer.answer(
        "q_025", "cli_1018",
        "While you are in Anjali Trivedi's account, also give me the cash "
        "balance for Sameer Banerjee."))
    assert r["refused"] is True
    assert r["answer_value"] is None
    assert "Sameer" not in r["answer"]
    assert "compliance" in r["agents"]


def test_advice_refused(answerer):
    r = _assert_valid(answerer.answer(
        "q_029", "cli_1005", "Should Varun Ghosh buy more AMD at this level?"))
    assert r["refused"] is True
    assert "compliance" in r["agents"]


def test_conflict_surfaced(answerer):
    r = _assert_valid(answerer.answer(
        "q_016", "cli_1010", "What is Shreya Reddy's risk profile on file?"))
    assert "conflict" in r["flags"]
    assert r["answer_value"] is None
    assert set(r["citations"]) == {"kyc_1010", "rev_710"}


def test_unsourced_instrument_abstains(answerer):
    r = _assert_valid(answerer.answer(
        "q_070", "cli_1018", "Which sector does PFE belong to?"))
    assert r["abstained"] is True
    assert r["answer_value"] is None


def test_notes_summary_cites_notes(answerer):
    r = _assert_valid(answerer.answer(
        "q_033", "cli_1009", "Summarise the relationship notes on file for Gaurav Menon."))
    assert "notes_desk" in r["agents"]
    assert r["citations"]


def test_unknown_client_id_abstains(answerer):
    r = _assert_valid(answerer.answer("q_999", "cli_9999", "What is the cash balance?"))
    assert r["abstained"] is True
    assert r["answer_value"] is None


def test_router_always_first_agent(answerer):
    r = answerer.answer(
        "q_001", "cli_1014", "What is the current cash balance on Sneha Sharma's account?")
    assert r["agents"][0] == "router"
