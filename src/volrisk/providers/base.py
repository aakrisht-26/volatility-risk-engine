"""Abstract interface for daily OHLCV data providers.

The rest of the pipeline depends only on this interface and its canonical output
schema. The primary provider (yfinance) is an unofficial API that occasionally
breaks; isolating it here means a provider swap never touches ingestion,
validation, or modeling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

#: Canonical column order for daily bars everywhere downstream.
#: ``trade_date`` is the exchange-local trading day (a DATE, no intraday time).
#: Both adjusted and unadjusted close are kept.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


class OHLCVProvider(ABC):
    """A source of daily OHLCV bars in the canonical schema."""

    @abstractmethod
    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch daily bars for ``ticker`` between ``start`` and ``end`` (inclusive).

        Implementations must return a DataFrame with exactly ``CANONICAL_COLUMNS``:
        ``trade_date`` as ``datetime.date``, prices as float64, ``volume`` as
        nullable Int64, sorted by ``trade_date`` with no duplicate
        (ticker, trade_date) pairs. Raise ``ValueError`` if the provider returns
        no usable data.
        """
