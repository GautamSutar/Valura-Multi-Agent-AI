"""Validates a response dict against schema/answer.schema.json before it
leaves the service. A response that fails this never goes out: the
orchestrator falls back to a minimal, definitely-valid abstention instead."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "answer.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


def errors(response: dict) -> list[str]:
    return [e.message for e in _VALIDATOR.iter_errors(response)]


def is_valid(response: dict) -> bool:
    return not errors(response)
