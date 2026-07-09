"""Tests for the pandera daily-bars schema. Fixture-driven — no network.

The corrupt fixture violates one rule per row: high < low (2024-01-03),
negative open (2024-01-04), null close (2024-01-05), and a duplicate
(ticker, trade_date) pair (2024-01-08).
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from volrisk.validate.schemas import validate_daily_bars

FIXTURES = Path(__file__).parent / "fixtures"


def canonical_frame(ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [187.15, 184.22],
            "high": [188.44, 185.88],
            "low": [183.885, 183.43],
            "close": [185.64, 184.25],
            "adj_close": [184.55, 183.17],
            "volume": pd.array([82488700, 58414500], dtype="Int64"),
        }
    )


def load_corrupt_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "canonical_daily_corrupt.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).map(lambda ts: ts.date())
    df["volume"] = df["volume"].astype("Int64")
    for col in ("open", "high", "low", "close", "adj_close"):
        df[col] = df[col].astype("float64")
    return df


def test_valid_batch_passes_unchanged() -> None:
    df = canonical_frame()
    out = validate_daily_bars(df)
    pd.testing.assert_frame_equal(out, df)


def test_zero_and_missing_volume_are_allowed() -> None:
    df = canonical_frame("^VIX")
    df["volume"] = pd.array([0, None], dtype="Int64")
    validate_daily_bars(df)


def test_corrupted_fixture_reports_every_violation() -> None:
    with pytest.raises(SchemaErrors) as excinfo:
        validate_daily_bars(load_corrupt_fixture(), context="corrupt-fixture")

    failures = excinfo.value.failure_cases
    checks = " | ".join(failures["check"].astype(str).unique())

    assert "high_ge_low" in checks  # 2024-01-03: high 183.00 < low 185.88
    assert "greater_than(0)" in checks  # 2024-01-04: open is negative
    assert "not_nullable" in checks  # 2024-01-05: close is null
    assert "uniqueness" in checks  # 2024-01-08 appears twice


def test_single_corrupt_row_fails_whole_batch() -> None:
    df = canonical_frame()
    df.loc[1, "high"] = df.loc[1, "low"] / 2  # break high >= low on one row only

    with pytest.raises(SchemaErrors):
        validate_daily_bars(df)


def test_unexpected_extra_column_is_rejected() -> None:
    df = canonical_frame().assign(signal=1.0)  # strict schema: canonical columns only

    with pytest.raises(SchemaErrors):
        validate_daily_bars(df)


def test_missing_column_is_rejected() -> None:
    df = canonical_frame().drop(columns=["volume"])

    with pytest.raises(SchemaErrors):
        validate_daily_bars(df)


def test_intraday_timestamps_are_rejected() -> None:
    df = canonical_frame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])  # datetimes, not dates

    with pytest.raises(SchemaErrors):
        validate_daily_bars(df)
