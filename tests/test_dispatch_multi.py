"""Exercises the multi-agent path (two facts, two roles) directly against
`dispatch.resolve`, and the blackout upstream-issue path directly against
`llmclient.call_agent`, without depending on a particular router
classification -- the mock reasoning gateway's classifier is too crude to
reliably produce a `secondary_intent`, so these are pinned at the layer
below it instead.
"""
from types import SimpleNamespace

from takehome import dispatch
from takehome.data import load_book_and_market
from takehome.llmclient import call_agent


def _cls(**kw):
    defaults = dict(roles=[], intent="other", secondary_intent=None,
                    symbol=None, txn_type=None, txn_id=None, year=None,
                    month=None, date_from=None, date_to=None, asof=None,
                    kyc_field=None, sector=None, cross_client_attempt=False,
                    other_client_ref=None, advice_request=False)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_multi_agent_pan_and_first_buy():
    book, market = load_book_and_market("data/client_book.json", "data/market_data.json")
    client = book.client("cli_1014")
    cls = _cls(intent="kyc_field", kyc_field="pan",
              secondary_intent="first_txn_date", symbol="AAPL", txn_type="buy")
    primary = dispatch.resolve(book, market, client, cls.intent, cls)
    secondary = dispatch.resolve(book, market, client, cls.secondary_intent, cls)
    assert primary.role == "kyc_profile"
    assert primary.result.value == "****249H"
    assert secondary.role == "book_qa"
    assert secondary.result.found


def test_multi_agent_notes_and_cash():
    book, market = load_book_and_market("data/client_book.json", "data/market_data.json")
    client = book.client("cli_1014")
    cls = _cls(intent="notes_summary", secondary_intent="cash_balance")
    secondary = dispatch.resolve(book, market, client, cls.secondary_intent, cls)
    assert secondary.role == "book_qa"
    assert secondary.result.value is not None


class _ErroredResult:
    status = None  # set in test
    content = ("You exceeded your current quota. The upstream is unavailable "
              "for the remainder of this outage.")


class _FakeAgent:
    name = "router"

    def __init__(self, status):
        self._status = status

    def run(self, *a, **kw):
        from agno.run.base import RunStatus
        r = _ErroredResult()
        r.status = self._status
        return r


def test_blackout_sets_upstream_issue_flag():
    from agno.run.base import RunStatus
    outcome = call_agent(_FakeAgent(RunStatus.error), "hi")
    assert outcome.ok is False
    assert outcome.upstream_issue is True
