from __future__ import annotations

from decimal import Decimal

import pytest

from obspipe.parsing import (
    MONEY_SCAN_RE,
    appears_in,
    is_valid_iso_date,
    parse_date,
    parse_money,
)


@pytest.mark.parametrize(
    ("raw", "value", "currency"),
    [
        ("$1,250.00", Decimal("1250.00"), "USD"),
        ("EUR 2.500,75", Decimal("2500.75"), "EUR"),
        ("€1.234.567,89", Decimal("1234567.89"), "EUR"),
        ("Rs. 3,50,000", Decimal("350000"), "INR"),
        ("₹45,000", Decimal("45000"), "INR"),
        ("45 000 USD", Decimal("45000"), "USD"),
        ("US$12.5", Decimal("12.5"), "USD"),
        ("CHF 90", Decimal("90"), "CHF"),
        ("1250", Decimal("1250"), None),
        ("($300.00)", Decimal("-300.00"), "USD"),
    ],
)
def test_parse_money_handles_locale_and_symbol_variants(raw, value, currency):
    assert parse_money(raw) == (value, currency)


def test_parse_money_reports_missing_pieces_instead_of_defaulting():
    assert parse_money("no money here") == (None, None)
    # A bare symbol yields a currency but no number - never a zero.
    assert parse_money("$") == (None, "USD")


@pytest.mark.parametrize(
    ("raw", "iso"),
    [
        ("2024-05-02", "2024-05-02"),
        ("March 3, 2024", "2024-03-03"),
        ("15 January 2024", "2024-01-15"),
        ("01.03.2024", "2024-03-01"),
        ("gibberish", None),
        ("0l March 2024", None),
    ],
)
def test_parse_date(raw, iso):
    assert parse_date(raw) == iso


def test_is_valid_iso_date_rejects_impossible_dates():
    assert is_valid_iso_date("2024-02-29")
    assert not is_valid_iso_date("2023-02-29")
    assert not is_valid_iso_date("not-a-date")
    assert not is_valid_iso_date(None)


def test_appears_in_ignores_whitespace_and_case():
    assert appears_in("Acme  Corp", "the ACME corp signed")
    assert not appears_in("Globex", "the ACME corp signed")
    assert not appears_in("   ", "anything")


def test_money_scanner_does_not_span_newlines():
    """A cross-line match would make grounding checks fire on invented spans."""
    assert MONEY_SCAN_RE.findall("date 2024-05-02\nUSD") == []
    assert len(MONEY_SCAN_RE.findall("pay $10.00 and €20,00 today")) == 2
