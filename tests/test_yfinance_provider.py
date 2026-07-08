"""Offline tests for yfinance frame normalization. Fixture-driven — no network.

The fixture mimics the raw shape of a yfinance daily download and deliberately
contains a duplicate date, an all-NaN row, and a missing volume so the
normalization contract is pinned by tests.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volrisk.providers.base import CANONICAL_COLUMNS
from volrisk.providers.yfinance_provider import normalize_yfinance_frame

FIXTURES = Path(__file__).parent / "fixtures"


def load_raw_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "yfinance_daily_sample.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def test_normalize_returns_canonical_schema() -> None:
    out = normalize_yfinance_frame(load_raw_fixture(), "AAPL")

    assert list(out.columns) == list(CANONICAL_COLUMNS)
    for col in ("open", "high", "low", "close", "adj_close"):
        assert out[col].dtype == "float64", col
    assert out["volume"].dtype == "Int64"
    assert (out["ticker"] == "AAPL").all()


def test_trade_dates_are_python_dates_and_sorted() -> None:
    out = normalize_yfinance_frame(load_raw_fixture(), "AAPL")

    assert all(isinstance(d, date) for d in out["trade_date"])
    assert out["trade_date"].is_monotonic_increasing


def test_all_nan_ohlc_row_is_dropped() -> None:
    out = normalize_yfinance_frame(load_raw_fixture(), "AAPL")

    assert date(2024, 1, 8) not in set(out["trade_date"])
    assert len(out) == 4


def test_duplicate_trade_date_keeps_last_row() -> None:
    out = normalize_yfinance_frame(load_raw_fixture(), "AAPL")

    dup = out[out["trade_date"] == date(2024, 1, 5)]
    assert len(dup) == 1
    assert dup["close"].iloc[0] == 181.20
    assert dup["volume"].iloc[0] == 62303400


def test_missing_volume_survives_as_na() -> None:
    out = normalize_yfinance_frame(load_raw_fixture(), "AAPL")

    row = out[out["trade_date"] == date(2024, 1, 4)]
    assert row["volume"].isna().all()


def test_multiindex_columns_are_flattened() -> None:
    raw = load_raw_fixture()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])

    out = normalize_yfinance_frame(raw, "AAPL")

    assert list(out.columns) == list(CANONICAL_COLUMNS)
    assert len(out) == 4


def test_timezone_aware_index_is_reduced_to_local_date() -> None:
    raw = load_raw_fixture()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize("America/New_York")

    out = normalize_yfinance_frame(raw, "AAPL")

    assert out["trade_date"].iloc[0] == date(2024, 1, 2)


def test_missing_expected_column_raises() -> None:
    raw = load_raw_fixture().drop(columns=["Adj Close"])

    with pytest.raises(ValueError, match="Adj Close"):
        normalize_yfinance_frame(raw, "AAPL")


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_yfinance_frame(pd.DataFrame(), "AAPL")
