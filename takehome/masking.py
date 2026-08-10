"""Identity and bank-account masking. Deterministic, applied in code, so no
LLM response can bypass it: the mask is computed here and the raw value never
enters a prompt or leaves this module.
"""
from __future__ import annotations

MASK_PREFIX = "****"


def mask_tail(value: str) -> str:
    """Four asterisks followed by the last four characters, e.g. ****234F."""
    v = str(value)
    return MASK_PREFIX + v[-4:] if len(v) >= 4 else MASK_PREFIX + v


def mask_pan(pan: str) -> str:
    return mask_tail(pan)


def mask_bank_account(account_number: str) -> str:
    return mask_tail(account_number)
