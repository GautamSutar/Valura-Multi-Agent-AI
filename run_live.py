#!/usr/bin/env python
"""Runs this ecosystem against the real assessment server
(https://ai-arena.twocc.in), for practice, qualifying or final attempts.

This is the client-loop side of the submission: it fetches the book and
market once, declares the roster, then hands the protocol loop to
`harness/reference_client.py` (unmodified) with this ecosystem's
`takehome.ref_service.Reference` as the answerer. The HTTP service in
`takehome/service.py` is the other side of the same core -- used for local
practice against the bundled gateway and for the Docker packaging contract.

  python run_live.py --mode practice
  python run_live.py --mode qualifying
  python run_live.py --mode final

Reads ASSESSMENT_URL / ASSESSMENT_KEY from the environment (falls back to
VALURA_BASE_URL / VALURA_API_KEY, which is what this repo's .env uses during
development).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "harness"))

from reference_client import Client, run  # noqa: E402
import score as S  # noqa: E402

from takehome.ref_service import Reference  # noqa: E402

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("ASSESSMENT_URL")
                    or os.environ.get("VALURA_BASE_URL"))
    ap.add_argument("--key", default=os.environ.get("ASSESSMENT_KEY")
                    or os.environ.get("VALURA_API_KEY"))
    ap.add_argument("--mode", default="practice",
                    choices=["practice", "qualifying", "final"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not a.url or not a.key:
        raise SystemExit("need --url/--key, or ASSESSMENT_URL/ASSESSMENT_KEY "
                         "(or VALURA_BASE_URL/VALURA_API_KEY) in the environment")

    c = Client(a.url, a.key, a.mode)
    book = c.book()
    market = c.market()
    ref = Reference(book, c.llm_base_url(), api_key=a.key, market=market)

    me = run(a.url, a.key, a.mode, ref, a.quiet)
    print(json.dumps({k: v for k, v in me.items() if k != "scorecard"}, indent=1))
    if me.get("scorecard"):
        print(S.render(me["scorecard"]))


if __name__ == "__main__":
    main()
