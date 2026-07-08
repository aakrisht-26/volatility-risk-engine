"""Backfill daily OHLCV history into the raw parquet landing zone.

Usage::

    uv run python -m volrisk.ingest.backfill [--years 10] [--out data/raw] [--tickers ...]

The landing zone is immutable raw data: one parquet file per ticker, replayable
into Postgres at any time. A re-run replaces each file wholesale via an atomic
rename, so backfill is idempotent — duplicate rows are impossible by construction.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from volrisk.providers.base import OHLCVProvider
from volrisk.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

#: Phase-1 US basket. ^VIX is reference data (implied-vol comparison later);
#: it is never modeled as a tradable asset.
PHASE_1_TICKERS: tuple[str, ...] = (
    "^GSPC",
    "^VIX",
    "AAPL",
    "MSFT",
    "NVDA",
    "JPM",
    "XOM",
    "TSLA",
)

DEFAULT_YEARS = 10
DEFAULT_OUT_DIR = Path("data/raw")


def parquet_path(out_dir: Path, ticker: str) -> Path:
    """Filesystem-friendly parquet path for a ticker (index caret stripped: ^GSPC -> GSPC).

    The true symbol is preserved in the file's ``ticker`` column; the filename is
    only a convenience.
    """
    return out_dir / f"{ticker.replace('^', '')}.parquet"


def backfill_ticker(
    provider: OHLCVProvider, ticker: str, start: date, end: date, out_dir: Path
) -> int:
    """Fetch bars for one ticker and atomically (re)write its parquet. Returns row count."""
    df = provider.fetch_daily_ohlcv(ticker, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = parquet_path(out_dir, ticker)
    tmp = target.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)  # atomic on the same filesystem; no half-written landing files
    logger.info("%s: wrote %d rows -> %s", ticker, len(df), target)
    return len(df)


def run_backfill(
    provider: OHLCVProvider,
    tickers: tuple[str, ...] = PHASE_1_TICKERS,
    years: int = DEFAULT_YEARS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, int]:
    """Backfill ``years`` of daily bars for every ticker. Returns per-ticker row counts."""
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=years)).date()
    counts: dict[str, int] = {}
    for ticker in tickers:
        counts[ticker] = backfill_ticker(provider, ticker, start, end, out_dir)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill daily OHLCV to raw parquet.")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tickers", nargs="+", default=list(PHASE_1_TICKERS))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    counts = run_backfill(YFinanceProvider(), tuple(args.tickers), args.years, args.out)

    # CLI entrypoint summary (print is fine here; library code logs only).
    print(f"\nBackfill complete: {args.years}y of daily bars through {date.today().isoformat()}")
    for ticker, n in counts.items():
        print(f"  {ticker:>8}  {n:6d} rows")
    print(f"  {'TOTAL':>8}  {sum(counts.values()):6d} rows")


if __name__ == "__main__":
    main()
