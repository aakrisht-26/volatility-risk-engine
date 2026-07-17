"""Offline tests for the Stooq fallback provider and the fallback chain.

The adjusted-only policy is the load-bearing contract: Stooq rows must carry
close == adj_close so the source flag (raw.daily_bars.source = 'stooq') is the
ONLY thing separating them from dual-price yfinance rows.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volrisk.providers.base import CANONICAL_COLUMNS, FallbackProvider, OHLCVProvider
from volrisk.providers.stooq_provider import normalize_stooq_frame, stooq_symbol

FIXTURES = Path(__file__).parent / "fixtures"


def load_raw_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "stooq_daily_sample.csv")


# --- symbol mapping ---


def test_us_equities_map_to_lowercase_dot_us() -> None:
    assert stooq_symbol("AAPL") == "aapl.us"
    assert stooq_symbol("TSLA") == "tsla.us"


def test_indices_use_explicit_mappings() -> None:
    assert stooq_symbol("^GSPC") == "^spx"
    assert stooq_symbol("^VIX") == "^vix"


def test_unmapped_index_raises() -> None:
    with pytest.raises(ValueError, match="no Stooq symbol mapping"):
        stooq_symbol("^NSEI")


# --- normalization and the adjusted-only policy ---


def test_normalize_returns_canonical_schema_sorted() -> None:
    out = normalize_stooq_frame(load_raw_fixture(), "AAPL")

    assert list(out.columns) == list(CANONICAL_COLUMNS)
    assert (out["ticker"] == "AAPL").all()
    assert out["trade_date"].is_monotonic_increasing
    assert all(isinstance(d, date) for d in out["trade_date"])
    assert out["volume"].dtype == "Int64"
    assert out["volume"].isna().iloc[-1]  # blank volume survives as NA


def test_adjusted_only_policy_close_equals_adj_close() -> None:
    out = normalize_stooq_frame(load_raw_fixture(), "AAPL")

    assert (out["close"] == out["adj_close"]).all()


def test_missing_column_raises() -> None:
    raw = load_raw_fixture().drop(columns=["Close"])

    with pytest.raises(ValueError, match="Close"):
        normalize_stooq_frame(raw, "AAPL")


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_stooq_frame(pd.DataFrame(), "AAPL")


# --- fallback chain ---


class Failing(OHLCVProvider):
    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raise ValueError(f"{ticker}: primary down")


class Serving(OHLCVProvider):
    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return normalize_stooq_frame(load_raw_fixture(), ticker)


def test_fallback_uses_secondary_and_records_source() -> None:
    provider = FallbackProvider([("yfinance", Failing()), ("stooq", Serving())])

    out = provider.fetch_daily_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))

    assert len(out) == 4
    assert provider.sources["AAPL"] == "stooq"


def test_fallback_prefers_primary_when_it_works() -> None:
    provider = FallbackProvider([("yfinance", Serving()), ("stooq", Failing())])

    provider.fetch_daily_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))

    assert provider.sources["AAPL"] == "yfinance"


def test_fallback_raises_when_every_provider_fails() -> None:
    provider = FallbackProvider([("yfinance", Failing()), ("stooq", Failing())])

    with pytest.raises(RuntimeError, match="every provider failed"):
        provider.fetch_daily_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))
