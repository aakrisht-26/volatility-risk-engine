"""Stooq CSV fallback provider (Step 11, triggered by the option-(a) ruling).

Stooq's daily CSV endpoint needs no cookies or auth, which makes it immune to
the fc.yahoo.com TLS-edge lottery and friendly to datacenter IPs — exactly the
failure modes that motivate a fallback on cold GitHub Actions runners.

Adjusted-only policy (recorded Step-11 ruling): Stooq serves ADJUSTED prices
only (splits and dividends folded in). Fallback rows therefore carry
close := adjusted close and adj_close := the same value — close == adj_close
by construction — and are flagged with raw.daily_bars.source = 'stooq' so
unadjusted-close consumers can filter them out. The modeling path (returns on
adj_close) is unaffected.

Symbol mapping: US equities map to "<lower>.us" (AAPL -> aapl.us); indices use
Stooq's own codes (^GSPC -> ^spx, ^VIX -> ^vix).
"""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

from volrisk.providers.base import CANONICAL_COLUMNS, OHLCVProvider

logger = logging.getLogger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/"

#: Indices need explicit mappings; US equities default to "<lower>.us".
_INDEX_SYMBOLS = {"^GSPC": "^spx", "^VIX": "^vix"}

_REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close")


def stooq_symbol(ticker: str) -> str:
    """Map a Yahoo-style ticker to Stooq's symbol scheme."""
    if ticker in _INDEX_SYMBOLS:
        return _INDEX_SYMBOLS[ticker]
    if ticker.startswith("^"):
        raise ValueError(f"no Stooq symbol mapping for index {ticker!r}")
    return f"{ticker.lower()}.us"


def normalize_stooq_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Stooq CSV -> canonical frame; close := adj_close := Stooq's adjusted close."""
    if raw.empty:
        raise ValueError(f"{ticker}: Stooq returned an empty frame")
    missing = [c for c in _REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{ticker}: Stooq frame is missing expected columns {missing}")

    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(raw["Date"]).map(lambda ts: ts.date()),
            "open": raw["Open"].astype("float64"),
            "high": raw["High"].astype("float64"),
            "low": raw["Low"].astype("float64"),
            "close": raw["Close"].astype("float64"),
            "adj_close": raw["Close"].astype("float64"),  # adjusted-only policy
        }
    )
    volume = raw["Volume"] if "Volume" in raw.columns else pd.Series(pd.NA, index=raw.index)
    df["volume"] = pd.to_numeric(volume, errors="coerce").round().astype("Int64")
    df["ticker"] = ticker
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    if df.empty:
        raise ValueError(f"{ticker}: no rows with price data after normalization")
    return (
        df[list(CANONICAL_COLUMNS)]
        .sort_values("trade_date")
        .drop_duplicates(subset=["ticker", "trade_date"], keep="last")
        .reset_index(drop=True)
    )


class StooqProvider(OHLCVProvider):
    """Fallback provider: Stooq daily CSV (adjusted prices; see module docstring)."""

    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "s": stooq_symbol(ticker),
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
        }
        logger.info("fetching %s daily bars %s..%s from Stooq", ticker, start, end)
        response = requests.get(STOOQ_URL, params=params, timeout=30)
        response.raise_for_status()
        body = response.text.strip()
        if not body or body.lower().startswith("no data"):
            raise ValueError(f"{ticker}: Stooq returned no data")
        return normalize_stooq_frame(pd.read_csv(io.StringIO(body)), ticker)
