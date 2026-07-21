"""Clean layer: align raw bars to the exchange calendar and load ``clean.daily_bars``.

Partial same-day bar policy (Step 5, recorded requirement)
    A bar enters ``clean`` only if its trade_date is a COMPLETED session of the
    ticker's exchange calendar at run time — the session's market_close must be
    <= the run's as-of timestamp (UTC). A fetch during market hours lands
    today's in-progress bar in ``raw``; it is excluded here and flows into
    ``clean`` on a later run, after the trailing-window re-fetch has revised it
    to final values. The modeling layer therefore never sees an in-progress bar.

Non-session rows
    Bars dated on days the calendar says the exchange was closed (e.g. an index
    printing on an equity holiday) are excluded from ``clean`` and reported.

Gap report
    Per ticker: expected sessions between its first bar and the last completed
    session, bars present, missing sessions, non-session rows, and excluded
    partial bars — reconciling exact per-ticker counts against the calendar.

Usage::

    uv run python -m volrisk.transform.cleaning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd
import pandas_market_calendars as mcal
from sqlalchemy import Engine

from volrisk.db.engine import get_engine
from volrisk.db.loaders import upsert_clean_daily_bars
from volrisk.transform.returns import add_log_returns, telescoping_check

logger = logging.getLogger(__name__)

#: The whole Phase-1 US basket aligns against XNYS. ^VIX is CBOE-published but
#: is ingested as reference data; its calendar quirks surface in the gap report.
EXCHANGE = "XNYS"

#: Telescoping identity tolerance: float64 rounding over ~2.5k daily sums.
TELESCOPE_TOLERANCE = 1e-9


@dataclass
class TickerGapReport:
    ticker: str
    first_bar: date
    last_bar: date
    expected_sessions: int
    bars_present: int
    missing_sessions: list[date] = field(default_factory=list)
    non_session_rows: list[date] = field(default_factory=list)
    partial_rows: list[date] = field(default_factory=list)
    clean_rows: int = 0


def last_completed_session(calendar, as_of: datetime) -> date:
    """Most recent session whose market_close is <= ``as_of`` (tz-aware UTC)."""
    sched = calendar.schedule(
        start_date=(as_of - pd.Timedelta(days=14)).date(), end_date=as_of.date()
    )
    done = sched[sched["market_close"] <= as_of]
    return done.index[-1].date()


def next_session(after: date, exchange: str = EXCHANGE) -> date:
    """First trading session strictly after ``after`` (weekends/holidays skipped)."""
    calendar = mcal.get_calendar(exchange)
    sched = calendar.schedule(
        start_date=after + pd.Timedelta(days=1), end_date=after + pd.Timedelta(days=14)
    )
    return sched.index[0].date()


def clean_ticker_frame(
    df: pd.DataFrame, calendar, as_of: datetime
) -> tuple[pd.DataFrame, TickerGapReport]:
    """Align one ticker's raw bars to the calendar; return (clean frame, report)."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    ticker = str(df["ticker"].iloc[0])
    first_bar = df["trade_date"].iloc[0]
    last_bar = df["trade_date"].iloc[-1]
    cutoff = last_completed_session(calendar, as_of)

    partial = sorted(d for d in df["trade_date"] if d > cutoff)
    window_end = min(last_bar, cutoff)
    schedule = calendar.schedule(start_date=first_bar, end_date=window_end)
    sessions = {ts.date() for ts in schedule.index}
    settled = set(df["trade_date"]) - set(partial)
    missing = sorted(sessions - settled)
    non_session = sorted(settled - sessions)

    keep = df[df["trade_date"].isin(sessions & settled)].reset_index(drop=True)
    keep = add_log_returns(keep)

    report = TickerGapReport(
        ticker=ticker,
        first_bar=first_bar,
        last_bar=last_bar,
        expected_sessions=len(sessions),
        bars_present=len(df),
        missing_sessions=missing,
        non_session_rows=non_session,
        partial_rows=partial,
        clean_rows=len(keep),
    )
    return keep, report


def run_cleaning(engine: Engine, as_of: datetime | None = None) -> list[TickerGapReport]:
    """Clean every ticker present in raw.daily_bars; upsert results, return reports."""
    as_of = as_of or datetime.now(UTC)
    calendar = mcal.get_calendar(EXCHANGE)
    raw = pd.read_sql_query(
        "SELECT ticker, trade_date, open, high, low, close, adj_close, volume"
        " FROM raw.daily_bars ORDER BY ticker, trade_date",
        engine,
    )
    if pd.api.types.is_datetime64_any_dtype(raw["trade_date"]):
        raw["trade_date"] = raw["trade_date"].map(lambda ts: ts.date())

    reports: list[TickerGapReport] = []
    for ticker, group in raw.groupby("ticker", sort=True):
        clean_df, report = clean_ticker_frame(group, calendar, as_of)
        upsert_clean_daily_bars(engine, clean_df, context=f"clean:{ticker}")
        logger.info(
            "%s: %d raw bars -> %d clean rows (missing=%d, non-session=%d, partial=%d)",
            ticker,
            report.bars_present,
            report.clean_rows,
            len(report.missing_sessions),
            len(report.non_session_rows),
            len(report.partial_rows),
        )
        reports.append(report)
    return reports


def _print_gap_report(reports: list[TickerGapReport], as_of: datetime, cutoff: date) -> None:
    print(f"\n=== Gap report ({EXCHANGE}) ===")
    print(f"as of {as_of:%Y-%m-%d %H:%M:%S %Z} | last completed session: {cutoff}")
    print(
        f"{'ticker':>7}  {'span':<24} {'sessions':>8} {'bars':>6} "
        f"{'missing':>7} {'non-sess':>8} {'partial':>7} {'clean':>6}"
    )
    for r in reports:
        print(
            f"{r.ticker:>7}  {r.first_bar}..{r.last_bar}  {r.expected_sessions:>8} "
            f"{r.bars_present:>6} {len(r.missing_sessions):>7} {len(r.non_session_rows):>8} "
            f"{len(r.partial_rows):>7} {r.clean_rows:>6}"
        )
    for r in reports:
        for label, dates in (
            ("missing sessions", r.missing_sessions),
            ("non-session rows (excluded)", r.non_session_rows),
            ("partial bars (excluded)", r.partial_rows),
        ):
            if dates:
                shown = ", ".join(str(d) for d in dates[:10]) + (" …" if len(dates) > 10 else "")
                print(f"  {r.ticker}: {label}: {shown}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    engine = get_engine()
    as_of = datetime.now(UTC)
    reports = run_cleaning(engine, as_of)
    cutoff = last_completed_session(mcal.get_calendar(EXCHANGE), as_of)
    _print_gap_report(reports, as_of, cutoff)

    print("\n=== Telescoping check: sum(log returns) vs ln(P_end/P_start) ===")
    clean_df = pd.read_sql_query(
        "SELECT ticker, trade_date, adj_close, log_return"
        " FROM clean.daily_bars ORDER BY ticker, trade_date",
        engine,
    )
    failures = 0
    for ticker, group in clean_df.groupby("ticker", sort=True):
        total, endpoints = telescoping_check(group)
        diff = abs(total - endpoints)
        verdict = "OK" if diff < TELESCOPE_TOLERANCE else "FAIL"
        failures += verdict == "FAIL"
        print(
            f"  {ticker:>7}  sum={total:+.10f}  ln(end/start)={endpoints:+.10f}"
            f"  |diff|={diff:.2e}  {verdict}"
        )
    if failures:
        raise SystemExit(f"{failures} ticker(s) failed the telescoping check")


if __name__ == "__main__":
    main()
