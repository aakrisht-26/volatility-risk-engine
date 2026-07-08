"""yfinance-backed OHLCV provider (primary source).

Network access is confined to :meth:`YFinanceProvider.fetch_daily_ohlcv`.
Parsing/normalization lives in the pure function :func:`normalize_yfinance_frame`
so it can be unit-tested offline against fixture data.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from yfinance.data import YfData

from volrisk.providers.base import CANONICAL_COLUMNS, OHLCVProvider

logger = logging.getLogger(__name__)

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
_PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


def normalize_yfinance_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a raw yfinance download frame into the canonical schema.

    Handles both flat and MultiIndex column layouts (yfinance returns
    ``(field, ticker)`` MultiIndex columns for some call shapes), strips any
    timezone/intraday component down to the exchange-local trade date, drops
    rows with no OHLC data at all, and deduplicates on (ticker, trade_date).
    Strict quality validation (nulls, price sanity) is the Step 3 pandera
    layer's job — this only guarantees shape and dtypes.
    """
    if raw.empty:
        raise ValueError(f"{ticker}: provider returned an empty frame")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [col for col in _COLUMN_MAP if col not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: provider frame is missing expected columns {missing}")

    df = df[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)
    df.columns.name = None  # yfinance labels the column index "Price"; drop the noise

    # Daily bars are exchange-local trading days: drop tz and time-of-day, keep the DATE.
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()

    df = df.dropna(subset=list(_PRICE_COLUMNS), how="all")
    if df.empty:
        raise ValueError(f"{ticker}: no rows with price data after normalization")

    df = df.astype(dict.fromkeys(_PRICE_COLUMNS, "float64"))
    df["volume"] = df["volume"].round().astype("Int64")

    df["ticker"] = ticker
    df["trade_date"] = df.index.date

    return (
        df.reset_index(drop=True)[list(CANONICAL_COLUMNS)]
        .sort_values("trade_date")
        .drop_duplicates(subset=["ticker", "trade_date"], keep="last")
        .reset_index(drop=True)
    )


class YFinanceProvider(OHLCVProvider):
    """Primary provider: Yahoo Finance daily bars via the yfinance package."""

    def __init__(self) -> None:
        # Yahoo's fc.yahoo.com cookie endpoint serves a TLS certificate that fails
        # hostname verification (observed 2026-07), and yfinance's default "basic"
        # cookie strategy propagates that error instead of falling back. Force the
        # "csrf" strategy (guce.yahoo.com consent flow), which verifies cleanly.
        # Private yfinance API — acceptable only inside this provider module, whose
        # whole purpose is to quarantine yfinance instability.
        YfData()._set_cookie_strategy("csrf")

    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        logger.info("fetching %s daily bars %s..%s from yfinance", ticker, start, end)
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance treats end as exclusive
            interval="1d",
            auto_adjust=False,  # keep both adjusted and unadjusted close
            actions=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"{ticker}: yfinance returned no data")
        return normalize_yfinance_frame(raw, ticker)
