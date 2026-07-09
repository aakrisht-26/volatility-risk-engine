"""Offline tests for yfinance frame normalization. Fixture-driven — no network.

The fixture mimics the raw shape of a yfinance daily download and deliberately
contains a duplicate date, an all-NaN row, and a missing volume so the
normalization contract is pinned by tests.
"""

from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

import volrisk.providers.yfinance_provider as yfp
from volrisk.providers.base import CANONICAL_COLUMNS
from volrisk.providers.yfinance_provider import (
    YFinanceProvider,
    is_certificate_error,
    normalize_yfinance_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"

CERT_ERROR = Exception(
    "Failed to perform, curl: (60) SSL: no alternative certificate subject name "
    "matches target hostname 'fc.yahoo.com'"
)


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


# --- cookie-strategy pinning and bounded retry (all offline, yfinance mocked) ---


class RecordingYfData:
    """Stands in for yfinance's YfData singleton; records strategy pins."""

    strategies: ClassVar[list[str]] = []

    def _set_cookie_strategy(self, strategy: str) -> None:
        RecordingYfData.strategies.append(strategy)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> YFinanceProvider:
    """A provider whose yfinance singleton and sleeps are neutered."""
    RecordingYfData.strategies = []
    monkeypatch.setattr(yfp, "YfData", RecordingYfData)
    monkeypatch.setattr(yfp.time, "sleep", lambda seconds: None)
    return YFinanceProvider()


def test_init_pins_csrf_cookie_strategy(provider: YFinanceProvider) -> None:
    assert RecordingYfData.strategies == ["csrf"]
    assert yfp.yf.config.debug.hide_exceptions is False


def test_certificate_error_detected_through_exception_chain() -> None:
    wrapper = ValueError("^GSPC: yfinance returned no data")
    wrapper.__cause__ = CERT_ERROR
    assert is_certificate_error(wrapper)
    assert is_certificate_error(CERT_ERROR)
    assert not is_certificate_error(ValueError("plain failure"))
    assert not is_certificate_error(None)


def test_fetch_retries_certificate_error_then_succeeds(
    provider: YFinanceProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def flaky_download(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls.append(1)
        if len(calls) == 1:
            raise CERT_ERROR
        return load_raw_fixture()

    monkeypatch.setattr(provider, "_download_raw", flaky_download)

    out = provider.fetch_daily_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))

    assert len(calls) == 2
    assert list(out.columns) == list(CANONICAL_COLUMNS)
    assert len(out) == 4
    # csrf re-pinned before every attempt (init + attempt 1 + attempt 2),
    # since yfinance can revert the strategy mid-flight.
    assert RecordingYfData.strategies == ["csrf", "csrf", "csrf"]


def test_fetch_gives_up_after_bounded_attempts(
    provider: YFinanceProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def always_cert_error(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls.append(1)
        raise CERT_ERROR

    monkeypatch.setattr(provider, "_download_raw", always_cert_error)

    with pytest.raises(ValueError, match="giving up after 3 fetch attempts") as excinfo:
        provider.fetch_daily_ohlcv("^GSPC", date(2024, 1, 1), date(2024, 1, 31))

    assert len(calls) == provider._MAX_ATTEMPTS
    assert excinfo.value.__cause__ is CERT_ERROR  # root cause preserved for the logs


def test_fetch_retries_empty_frames_then_gives_up(
    provider: YFinanceProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def empty_download(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame()

    monkeypatch.setattr(provider, "_download_raw", empty_download)

    with pytest.raises(ValueError, match="giving up"):
        provider.fetch_daily_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))

    assert len(calls) == provider._MAX_ATTEMPTS
