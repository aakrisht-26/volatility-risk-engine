"""yfinance-backed OHLCV provider (primary source).

Network access is confined to :class:`YFinanceProvider`. Parsing/normalization
lives in the pure function :func:`normalize_yfinance_frame` so it can be
unit-tested offline against fixture data.

Reliability notes (root-caused 2026-07-09, yfinance 1.5.1):

- Yahoo's ``fc.yahoo.com`` cookie endpoint serves a TLS certificate that fails
  hostname verification on a *subset* of Yahoo's edge nodes, so cold-start
  fetches fail intermittently depending on which edge a request lands on.
- Pinning yfinance's alternate "csrf" cookie strategy does NOT avoid this on
  its own: yfinance auto-reverts csrf -> basic whenever the csrf consent flow
  fails to mint a crumb (``data.py::_get_cookie_and_crumb``), and that flow
  reliably fails outside the EU ("Failed to find csrfToken in response").
- Once any fetch succeeds, yfinance persists a ~1-year cookie to its disk
  cache and later processes never contact a cookie endpoint — which makes the
  failure look intermittent across machines and days.

Deterministic handling: pin csrf at init (cheap best-effort bypass of the bad
endpoint), surface real fetch exceptions via the public
``yf.config.debug.hide_exceptions`` flag, and wrap each fetch in a bounded
retry that re-pins the strategy and re-rolls the edge-node dice. All of this
is quarantined here by design — yfinance instability must never leak past the
provider interface.
"""

from __future__ import annotations

import logging
import time
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
    """Convert a raw yfinance daily frame into the canonical schema.

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


def is_certificate_error(exc: BaseException | None) -> bool:
    """Walk an exception chain looking for a TLS certificate-verification failure.

    Matched by name/message rather than exception type so we stay decoupled
    from yfinance's transport library (currently curl_cffi, historically
    requests — both spell it differently).
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if "certificate" in text or "ssl" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class YFinanceProvider(OHLCVProvider):
    """Primary provider: Yahoo Finance daily bars via the yfinance package."""

    _MAX_ATTEMPTS = 3
    _RETRY_WAIT_SECONDS = (2.0, 5.0)

    def __init__(self) -> None:
        # Surface real fetch exceptions instead of yfinance's default of logging
        # them and handing back an empty frame (public config flag).
        yf.config.debug.hide_exceptions = False
        self._pin_csrf_cookie_strategy()

    @staticmethod
    def _pin_csrf_cookie_strategy() -> None:
        """Point yfinance at the csrf cookie flow, away from broken fc.yahoo.com.

        Not sufficient alone — yfinance reverts csrf -> basic whenever the csrf
        flow fails (see module docstring); the bounded retry in
        ``fetch_daily_ohlcv`` is what makes fetches deterministic. Private
        yfinance API, acceptable only inside this module.
        """
        YfData()._set_cookie_strategy("csrf")

    def _download_raw(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Single network fetch; isolated so tests can substitute it."""
        return yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance treats end as exclusive
            interval="1d",
            auto_adjust=False,  # keep both adjusted and unadjusted close
            actions=False,
        )

    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            # Re-pin every attempt: yfinance may have reverted the strategy to
            # 'basic' during a previous attempt (singleton state).
            self._pin_csrf_cookie_strategy()
            try:
                logger.info(
                    "fetching %s daily bars %s..%s (attempt %d/%d)",
                    ticker,
                    start,
                    end,
                    attempt,
                    self._MAX_ATTEMPTS,
                )
                raw = self._download_raw(ticker, start, end)
                if raw is None or raw.empty:
                    raise ValueError(f"{ticker}: yfinance returned no data")
            except Exception as exc:  # every yfinance failure mode is retryable
                last_error = exc
                kind = (
                    "known fc.yahoo.com TLS-certificate failure"
                    if is_certificate_error(exc)
                    else "fetch error"
                )
                logger.warning(
                    "%s: attempt %d/%d failed (%s): %s",
                    ticker,
                    attempt,
                    self._MAX_ATTEMPTS,
                    kind,
                    exc,
                )
                if attempt < self._MAX_ATTEMPTS:
                    wait = self._RETRY_WAIT_SECONDS[
                        min(attempt - 1, len(self._RETRY_WAIT_SECONDS) - 1)
                    ]
                    time.sleep(wait)
                continue
            return normalize_yfinance_frame(raw, ticker)

        raise ValueError(
            f"{ticker}: giving up after {self._MAX_ATTEMPTS} fetch attempts"
        ) from last_error
