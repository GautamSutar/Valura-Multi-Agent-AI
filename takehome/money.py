"""Decimal helpers. All money and quantity arithmetic goes through here so
the same rounding rule applies everywhere an LLM never touches a figure."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

TWO_PLACES = Decimal("0.01")


def dec(value) -> Decimal:
    """Parse a book/market field into a Decimal. Raises on genuinely bad
    input rather than silently returning zero, so a malformed record surfaces
    instead of quietly miscounting."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def money2(value: Decimal) -> str:
    """Quantise to two decimal places, half-up, and render as a plain string
    (USD, no symbol, no thousands separator) per the response contract."""
    return str(value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def pct2(value: Decimal) -> str:
    return str(value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def qty_str(value: Decimal) -> str:
    """Quantities in this book carry 4 decimal places; preserve them exactly
    rather than re-quantising, since the key compares by decimal value, not
    string, but citations expect the same shape candidates saw in the data."""
    try:
        return str(value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return str(value)
