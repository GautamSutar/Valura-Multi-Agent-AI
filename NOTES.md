Email: gautamsutar.in@gmail.com

## Build and run

```bash
pip install -r requirements.txt
python -m pytest -q                      # unit + offline integration tests

# Local practice loop (stub gateway, no reasoning, exercises plumbing only):
python gateway/llm_gateway.py &
BOOK_PATH=data/client_book.json MARKET_PATH=data/market_data.json \
  LLM_BASE_URL=http://localhost:8600/v1 LLM_API_KEY=x PORT=8080 \
  python -m uvicorn takehome.service:app --port 8080 &
python harness/run_assessment.py --service http://localhost:8080 \
  --gateway http://localhost:8600 --questions questions/practice_questions.jsonl --out runs/latest
python harness/score.py --key harness/practice_key.json --leakmap harness/practice_leakmap.json \
  --transcript runs/latest/transcript.jsonl --usage runs/latest/gateway_usage.json --roster runs/latest/roster.json

# Real assessment (practice / qualifying / final):
ASSESSMENT_URL=https://ai-arena.twocc.in ASSESSMENT_KEY=vlr_... python run_live.py --mode practice

docker build -t valura-takehome .        # packaging contract; reads BOOK_PATH etc at runtime
```

## Architecture

`router` is a real Agno `Agent` (valura-fast) that classifies each question into a
structured `Classification` (an `intent`, and any symbol/date/txn-type it extracted)
via Agno's `output_schema`; the role is derived from `intent` in code
(`dispatch.INTENT_ROLE`), not asked of the model separately. `book_qa`, `kyc_profile`, `market_desk` and
`compliance` are also real `Agent` objects (declared in the roster, capable of running),
but their owned logic — retrieval, arithmetic, masking, the market-coverage boundary,
refusal wording — executes as deterministic Python in `takehome/retrieval.py` and
`dispatch.py`, because every figure here is checked exactly and an LLM cannot reliably
total a thousand transactions. `notes_desk` makes a second real model call to narrate
note/memo content, the one place free-text generation over untrusted record text is
actually needed. A deterministic `verifier` gates every response (owned citations,
re-masked PII) before it leaves the service. Cross-client and advice-solicitation
detection run as an **unconditional regex backstop**, ORed with the router's own
judgement — a misclassification there can disclose another client's data or give
advice, so it doesn't get to depend on the model alone. See `takehome/orchestrator.py`.

## Decisions made, not derived

- **Prose is templated, not model-generated**, for every intent except notes summaries.
  This trades a little polish in `judged_quality` (4 marks) for zero hallucination risk
  in the field graders check hardest (`grounded`, 24 marks) and near-zero token cost.
- **Blackout ⇒ abstain with `upstream_issue`**, rather than a rule-based fallback
  classifier. `score.py` marks an honest abstention exactly the same as a correct
  answer during blackout, so building a second, untested classification path bought
  nothing but risk.
- **Multi-agent questions use a `secondary_intent` field** on the same `Classification`
  rather than a list of sub-questions, since every multi-agent `kind` observed in
  practice is exactly two facts about one client.
- Would have asked: whether `kyc_field`-style questions for fields this book doesn't
  track (email, mobile, employer, nominee) are meant to score as `unanswerable`
  (my assumption, and what I implemented) or as a different category.

## The four questions

**How does it decide it cannot answer, and how is that not just the model unsure?**
Every `abstained=true` traces to a `QueryResult.found=False` from `retrieval.py` —
no deposits on record, an uncovered symbol, a KYC field this book doesn't track, no
price on or before the date. None of that is the model's confidence; it's a lookup
that came back empty. The router's own uncertainty is never consulted for this.

**A record instructs disclosure — where is that neutralised?**
Three layers, deliberately redundant. (1) The regex safety net in `safety.py` runs
before classification and doesn't read notes at all, so it can't be misled by their
content. (2) `notes_desk`'s system prompt states record text is data, never a
directive, and it never receives raw PAN/bank values to leak. (3) `verifier.py`
re-scans the *drafted answer text* for another client's name/PII regardless of which
path produced it, and swaps in a safe deterministic summary if it finds anything. For
this to reach the answer, the LLM would have to fabricate PII it was never given, and
the post-hoc scan would still have to miss the fabricated string.

**Provider down for an hour — what degrades?** Nothing gets *slower* in a useful
sense — `call_agent` retries up to 3 times over a fixed ~1.5s backoff (enough to ride
out the shared-capacity contention actually observed during qualifying attempts) and
then gives up well inside the 60s deadline. Every `book_qa`, `kyc_profile` and
`market_desk` answer is unaffected in *correctness*, since no model call is on that
path. `notes_desk` narration and cross-client/advice classification lose the model's
help but not their safety property: the regex backstop still refuses correctly without
the router. What genuinely degrades is coverage of ambiguously-phrased questions the
router can't classify without the model — those abstain honestly with `upstream_issue`
rather than guess, which costs marks on ordinary categories (only the officially
engineered `chaos_blackout` questions score an honest abstain the same as a right
answer) but never a fabrication or a leak.

**What did Agno make easy/hard, and what came from source, not docs?** Easy: wiring
`OpenAIChat` at an arbitrary `base_url` (the gateway) and getting structured Pydantic
output back via `output_schema` needed no special handling. Hard, and undocumented:
Agno sends the agent's `instructions` with role `"developer"`, not `"system"` — found
by pointing a raw HTTP logger at the gateway and reading the actual request body, not
by reading anything published. Also from source, not docs: on an upstream failure
Agno does **not** raise — `Agent.run()` returns a `RunOutput` with
`status=RunStatus.error` and the error text sitting in `.content`. My first blackout
run silently mis-flagged every blackout question as a generic internal error because
I was only catching `openai.RateLimitError`; `agno/agent/_response.py` was where I
found the real shape of an errored run, not an exception at all. A third finding, from
the first qualifying attempt against the real model rather than from source: Agno's
`output_schema` did not enforce a `Literal` field as strictly as I expected against this
provider — the model returned close paraphrases ("balance" for "cash_balance",
"purchase" for "buy") and pydantic rejected the *entire* `Classification` over one
field, not just that field. `intent` and `txn_type` are now plain strings, reconciled
against the canonical values in code (`dispatch.normalize_intent` /
`normalize_txn_type`) rather than trusted to arrive exact.

## Weak / next steps

**What I know from three qualifying attempts (30.3 → 32.3 → 31.9 / 96, all gates
passed, zero leaks/fabrication/PII across all three):** the roles-field and
strict-Literal fixes above came from reading these attempts' logs, in that order, and
each one was real but not the whole story — the third attempt's logs show the
remaining loss is dominated by genuine upstream capacity contention ("Token Plan rate
limit reached"), scattered across ordinary questions, not the engineered chaos bands.
The retry loop added afterward addresses that but was not itself validated against a
scored attempt, since all three were spent by the time it was built. This is the
single weakest point in the submission: the LLM-dependent path was validated
end-to-end against a mock reasoning gateway (`tests/mock_reasoning_gateway.py`) and,
after the fact, against the literal values a real model returned, but never against a
real model with the retry loop in place, because practice is always a non-reasoning
stub and qualifying attempts are finite.

The router uses one `intent` + optional `secondary_intent`; a question needing three
facts, or two facts from the *same* role, isn't representable. Given more time: a
genuine tool-calling notes_desk agent instead of a single narration call, and a small
cache keyed on `(client_id, intent, params)` for repeated/paraphrased questions — not
built, since the citation and value contract already makes correctness cheap without
one.
