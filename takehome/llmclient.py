"""Model construction and the resilience layer around every LLM call.

Every call this service makes goes through `call_agent`, which is where the
two chaos bands get handled: a bounded SDK-level retry absorbs the transient
band (the first call for a question fails, the next one succeeds), and a
short client-side timeout keeps a blackout from ever approaching the
60-second per-question deadline. Nothing here retries forever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import openai
from agno.agent import Agent
from agno.models.openai import OpenAIChat

logger = logging.getLogger("takehome.llm")

# One SDK-level retry: the gateway's transient band rejects exactly the first
# call for a question and lets every later one through, so a single retry is
# both necessary and sufficient. Retrying further only spends time a blackout
# will never reward.
SDK_MAX_RETRIES = 1
REQUEST_TIMEOUT_S = 12.0


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


def call_agent(agent: Agent, prompt: str, **run_kwargs) -> LLMOutcome:
    """Run an Agno agent and translate upstream failure into a flag the
    orchestrator can act on, instead of letting an exception reach the
    caller or a hung connection eat the question's deadline."""
    try:
        result = agent.run(prompt, **run_kwargs)
        return LLMOutcome(ok=True, content=result.content)
    except (openai.RateLimitError, openai.APIStatusError, openai.APITimeoutError,
            openai.APIConnectionError) as e:
        logger.warning("LLM call failed for agent %s: %s", agent.name, e)
        return LLMOutcome(ok=False, upstream_issue=True, detail=str(e))
    except Exception as e:  # noqa: BLE001 - never let a malformed model
        # response take the whole question down; fall back deterministically.
        logger.exception("unexpected error calling agent %s", agent.name)
        return LLMOutcome(ok=False, upstream_issue=False, detail=str(e))
