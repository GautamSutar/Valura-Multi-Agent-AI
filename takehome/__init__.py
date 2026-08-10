"""The candidate's take-home ecosystem.

Disables Agno's default telemetry call to os-api.agno.com at import time,
before any Agent is constructed: the grading network's only route out is the
LLM gateway, so that call would otherwise reach for the open internet and
either hang or fail on every single agent run.
"""
import os

os.environ.setdefault("AGNO_TELEMETRY", "false")
