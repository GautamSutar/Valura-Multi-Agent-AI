"""Model construction and the resilience layer around every LLM call.

Every call this service makes goes through `call_agent`. Two distinct kinds
of upstream trouble show up in practice: the assessment's own engineered
chaos (a clean transient band, then a clean blackout band, each covering a
contiguous run of questions) and genuine shared-capacity contention on the
real provider ("Token Plan rate limit reached"), observed scattered across
otherwise-normal questions when many candidates are running scored attempts
at once. The SDK's own single retry absorbs the former; it was not enough
for the latter, so `call_agent` adds its own short, bounded retry loop on
top. Nothing here retries forever -- the loop is capped well under the
60-second per-question deadline, and a genuine blackout (every call fails,
for a whole band) still gives up and abstains honestly rather than burn the
deadline chasing something that will not clear.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import openai
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.base import RunStatus

_UPSTREAM_MARKERS = ("quota", "rate limit", "rate_limit", "429",
                    "insufficient_quota", "unavailable", "token plan")

logger = logging.getLogger("takehome.llm")

SDK_MAX_RETRIES = 1
REQUEST_TIMEOUT_S = 12.0
# Total attempts made by call_agent's own loop, on top of the SDK's retry
# inside each attempt. Backoff is short and fixed, not exponential: the
# gateway's transient band clears on the very next call, and contention-based
# provider limits observed in practice cleared within a few seconds, so
# there is nothing to gain from a long climbing backoff -- only deadline
# budget to lose.
CALL_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.5


def make_model(tier: str, api_key: str, base_url: str) -> OpenAIChat:
    if tier not in ("valura-fast", "valura-deep"):
        raise ValueError(f"unknown model tier {tier!r}")
    return OpenAIChat(
        id=tier, api_key=api_key, base_url=base_url,
        timeout=REQUEST_TIMEOUT_S, max_retries=SDK_MAX_RETRIES,
    )


@dataclass
class LLMOutcome:
    ok: bool
    content: Any = None
    upstream_issue: bool = False
    detail: str = ""


def _call_once(agent: Agent, prompt: str, **run_kwargs) -> LLMOutcome:
    try:
        result = agent.run(prompt, **run_kwargs)
        if result.status == RunStatus.error:
            # Agno catches the upstream exception internally and returns an
            # errored RunOutput rather than raising: the exception handlers
            # below never fire for this case, so it has to be checked
            # explicitly. `content` is the error message text here, not an
            # answer, which is how a blackout call actually surfaces.
            text = str(result.content or "").lower()
            upstream = any(m in text for m in _UPSTREAM_MARKERS)
            logger.warning("agent %s run errored: %s", agent.name, result.content)
            return LLMOutcome(ok=False, upstream_issue=upstream, detail=text)
        return LLMOutcome(ok=True, content=result.content)
    except (openai.RateLimitError, openai.APIStatusError, openai.APITimeoutError,
            openai.APIConnectionError) as e:
        logger.warning("LLM call failed for agent %s: %s", agent.name, e)
        return LLMOutcome(ok=False, upstream_issue=True, detail=str(e))
    except Exception as e:  # noqa: BLE001 - never let a malformed model
        # response take the whole question down; fall back deterministically.
        logger.exception("unexpected error calling agent %s", agent.name)
        return LLMOutcome(ok=False, upstream_issue=False, detail=str(e))


def call_agent(agent: Agent, prompt: str, **run_kwargs) -> LLMOutcome:
    """Run an Agno agent with a short bounded retry loop, and translate a
    final upstream failure into a flag the orchestrator can act on, instead
    of letting an exception reach the caller or a hung connection eat the
    question's deadline."""
    outcome = LLMOutcome(ok=False)
    for attempt in range(CALL_ATTEMPTS):
        outcome = _call_once(agent, prompt, **run_kwargs)
        if outcome.ok:
            return outcome
        if not outcome.upstream_issue:
            # A malformed/unexpected response, not a capacity problem --
            # one retry can still help (a one-off glitch), but there is no
            # reason to spend the full budget on it.
            if attempt >= 1:
                break
        if attempt < CALL_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF_S)
    return outcome
