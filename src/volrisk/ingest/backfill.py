"""Backfill daily OHLCV history into the raw parquet landing zone.

Usage::

    uv run python -m volrisk.ingest.backfill [--start YYYY-MM-DD] [--out data/raw] [--tickers ...]

The landing zone is immutable raw data: one parquet file per ticker, replayable
into Postgres at any time. A re-run replaces each file wholesale via an atomic
rename, so backfill is idempotent — duplicate rows are impossible by construction.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from volrisk.providers.base import OHLCVProvider
from volrisk.providers.yfinance_provider import YFinanceProvider
from volrisk.validate.schemas import validate_daily_bars

logger = logging.getLogger(__name__)

#: Fixed data inception (audit fix 1, 2026-07-15). Originally "10 years back
#: from today", which made the window SLIDE: each day's backfill dropped the
#: oldest session from the parquet while the DB kept it, so the landing zone
#: stopped replaying the DB. Anchoring the start restores the invariant
#: per-ticker parquet rows == raw.daily_bars rows, and the replayability claim.
BACKFILL_START = date(2016, 7, 11)

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

DEFAULT_OUT_DIR = Path("data/raw")


def parquet_path(out_dir: Path, ticker: str) -> Path:
    """Filesystem-friendly parquet path for a ticker (index caret stripped: ^GSPC -> GSPC).

    The true symbol is preserved in the file's ``ticker`` column; the filename is
    only a convenience.
    """
    return out_dir / f"{ticker.replace('^', '')}.parquet"


@dataclass(frozen=True)
class BackfillTickerResult:
    ticker: str
    rows: int
    written: bool
    guarded: bool = False  # the monotonic guard refused a shrinking overwrite


def _existing_parquet_rows(target: Path) -> int:
    """Row count from parquet metadata — no data read."""
    return pq.ParquetFile(target).metadata.num_rows


def backfill_ticker(
    provider: OHLCVProvider,
    ticker: str,
    start: date,
    end: date,
    out_dir: Path,
    force: bool = False,
) -> BackfillTickerResult:
    """Fetch bars for one ticker and atomically (re)write its parquet.

    Monotonic landing-zone guard (Step-11 ruling): with an anchored inception,
    a legitimate refresh can only grow or match a ticker's history. A fresh
    fetch with FEWER rows than the existing file means the provider served a
    truncated series; the guard refuses the overwrite (ERROR log, ``guarded``
    result) so the caller can fall back to a trailing-window fetch.
    ``force=True`` overrides after human investigation.
    """
    df = provider.fetch_daily_ohlcv(ticker, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = parquet_path(out_dir, ticker)

    if target.exists() and not force:
        existing_rows = _existing_parquet_rows(target)
        if len(df) < existing_rows:
            logger.error(
                "%s: fresh fetch has %d rows < existing %d — monotonic guard refuses to "
                "shrink the landing zone (re-run with --force only after investigating)",
                ticker,
                len(df),
                existing_rows,
            )
            return BackfillTickerResult(ticker, rows=len(df), written=False, guarded=True)

    tmp = target.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)  # atomic on the same filesystem; no half-written landing files
    # Land first, validate second: the landing zone keeps exactly what the
    # provider returned (replayable, forensics), while a failed batch still
    # halts the pipeline here, before anything downstream consumes it.
    validate_daily_bars(df, context=ticker)
    logger.info("%s: wrote and validated %d rows -> %s", ticker, len(df), target)
    return BackfillTickerResult(ticker, rows=len(df), written=True)


def run_backfill(
    provider: OHLCVProvider,
    tickers: tuple[str, ...] = PHASE_1_TICKERS,
    start: date = BACKFILL_START,
    out_dir: Path = DEFAULT_OUT_DIR,
    force: bool = False,
) -> list[BackfillTickerResult]:
    """Backfill daily bars from the fixed inception for every ticker.

    The window's trailing edge is today; the leading edge never moves (see
    BACKFILL_START). Guarded tickers (monotonic guard) are returned unwritten
    for the caller to handle.
    """
    end = date.today()
    return [backfill_ticker(provider, t, start, end, out_dir, force=force) for t in tickers]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill daily OHLCV to raw parquet.")
    parser.add_argument("--start", type=date.fromisoformat, default=BACKFILL_START)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tickers", nargs="+", default=list(PHASE_1_TICKERS))
    parser.add_argument(
        "--force",
        action="store_true",
        help="override the monotonic landing-zone guard (only after investigating)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_backfill(
        YFinanceProvider(), tuple(args.tickers), args.start, args.out, force=args.force
    )

    # CLI entrypoint summary (print is fine here; library code logs only).
    print(f"\nBackfill complete: {args.start} (fixed inception) .. {date.today().isoformat()}")
    for res in results:
        marker = "  << GUARDED, not written" if res.guarded else ""
        print(f"  {res.ticker:>8}  {res.rows:6d} rows{marker}")
    print(f"  {'TOTAL':>8}  {sum(r.rows for r in results):6d} rows")
    if any(r.guarded for r in results):
        raise SystemExit("monotonic guard fired; investigate before using --force")


if __name__ == "__main__":
    main()
