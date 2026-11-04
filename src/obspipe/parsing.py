"""Deterministic parsers.

Everything here is pure Python on purpose. The LLM is allowed to *find* an
amount or a date; it is never allowed to decide what that amount or date
means numerically. That split is what makes step 2 failures diagnosable: if a
number is wrong, either the model quoted the wrong span (its fault) or the
parser mis-read the span (our fault), and the preserved ``raw_text`` tells you
which.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "US$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "Rs.": "INR",
    "Rs": "INR",
    "₩": "KRW",
    "₽": "RUB",
    "CHF": "CHF",
    "C$": "CAD",
    "A$": "AUD",
}

ISO_CODES = {
    "USD", "EUR", "GBP", "JPY", "INR", "CAD", "AUD", "CHF", "CNY", "SGD", "KRW", "RUB",
}

_AMOUNT_RE = re.compile(
    r"""
    (?P<prefix>US\$|C\$|A\$|Rs\.?|CHF|[A-Z]{3}\s|[$€£¥₹₩₽])?
    \s*
    (?P<number>
        \d{1,2}(?:,\d{2})+,\d{3}(?:\.\d{1,2})?
      | \d{1,3}(?:[,\s ]\d{3})+(?:[.,]\d{1,2})?
      | \d{1,3}(?:\.\d{3})+(?:,\d{1,2})?
      | \d+(?:[.,]\d+)?
    )
    (?!\d)
    \s*
    (?P<suffix>[A-Z]{3})?
    """,
    re.VERBOSE,
)

_EU_FORMAT_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?$|^\d+,\d{1,2}$")

_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%d.%m.%Y",
)


def normalize_currency_token(token: str | None) -> str | None:
    """Map a symbol or code to ISO-4217, or ``None`` if unrecognisable."""
    if not token:
        return None
    token = token.strip()
    if token.upper() in ISO_CODES:
        return token.upper()
    return CURRENCY_SYMBOLS.get(token) or CURRENCY_SYMBOLS.get(token.rstrip("."))


def parse_money(raw: str) -> tuple[Decimal | None, str | None]:
    """Parse ``raw`` into (value, ISO currency).

    Returns ``(None, currency)`` when a currency is identifiable but the number
    is not, and ``(value, None)`` for a bare number. Callers are expected to
    warn on either half being ``None`` rather than substituting a default.
    """
    if not raw:
        return None, None
    text = unicodedata.normalize("NFKC", raw).strip()
    match = _AMOUNT_RE.search(text)
    if not match:
        return None, normalize_currency_token(text)

    currency = normalize_currency_token(match.group("prefix")) or normalize_currency_token(
        match.group("suffix")
    )
    number = match.group("number")

    if _EU_FORMAT_RE.match(number):
        # 1.234,56 -> 1234.56
        number = number.replace(".", "").replace(",", ".")
    else:
        number = number.replace(",", "").replace(" ", "").replace(" ", "")

    try:
        value = Decimal(number)
    except InvalidOperation:
        value = None

    if value is not None and re.search(r"\(\s*[^)]*" + re.escape(match.group("number")), text):
        value = -value  # accounting-style negatives
    return value, currency


def numeric_span(raw: str) -> str | None:
    """The bare number inside an amount, with any currency marker removed.

    Lets a caller ask "is this figure in the document?" separately from "is this
    exact string in the document?" - the two differ whenever a table puts the
    currency in a column header and the model helpfully puts it back on the cell.
    """
    match = _AMOUNT_RE.search(unicodedata.normalize("NFKC", raw))
    return match.group("number") if match else None


def parse_date(raw: str) -> str | None:
    """Return an ISO-8601 date string, or ``None`` if unresolvable."""
    if not raw:
        return None
    text = unicodedata.normalize("NFKC", raw).strip().rstrip(".,;")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def is_valid_iso_date(value: str | None) -> bool:
    if not value:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _squash(text: str) -> str:
    return re.sub(r"[\s ]+", "", unicodedata.normalize("NFKC", text)).casefold()


def appears_in(needle: str, haystack: str) -> bool:
    """Whitespace- and case-insensitive containment check.

    Used to test whether a span the model claims to have extracted is really
    present in the source document. A miss is a hallucination signal.
    """
    if not needle.strip():
        return False
    return _squash(needle) in _squash(haystack)

MONEY_SCAN_RE = re.compile(
    r"(?:US\$|C\$|A\$|Rs\.?|CHF|[$€£¥₹₩₽]|\b(?:USD|EUR|GBP|JPY|INR|CAD|AUD|SGD)\b)"
    r"[ \t\u00a0]*\d[\d.,\u00a0 \t]*\d"
    r"|\d[\d.,\u00a0 \t]*\d[ \t\u00a0]*\b(?:USD|EUR|GBP|JPY|INR|CAD|AUD|SGD)\b"
)
"""Finds currency-marked amounts anywhere in a document.

Used as an independent second opinion on what the extractor should have found,
which is how a recall miss becomes detectable without ground-truth labels.
"""


_UNIT_RE = re.compile(
    r"[°%\u00b5\u2109\u2103]"
    r"|\u00d7\s*10|\bx\s*10"
    r"|\b(?:mmHg|mmhg|bpm|beats|min|sec|hrs?|hours?|days?|weeks?|months?|years?"
    r"|mg|kg|dL|mL|mmol|mol|IU|kPa|cm|mm|ppm|kcal)\b"
    r"|\b(?:g/dL|mg/dL|mEq/L|U/L)\b",
    re.IGNORECASE,
)


def looks_non_monetary(raw: str) -> bool:
    """True when a span is a measurement rather than a sum of money.

    Only consulted once :func:`parse_money` has already failed to find a
    currency, so a properly denominated amount can never be misread as a
    measurement. Catches vitals, lab values, rates and durations - the things a
    model will happily drop into an ``amounts`` field on a document that has no
    money in it at all.
    """
    return bool(_UNIT_RE.search(raw))
