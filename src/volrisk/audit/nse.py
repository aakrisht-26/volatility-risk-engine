"""India NSE data-quality audit — the Step-10 conditional gate.

Runs the three CLAUDE.md checks (gap rate vs the exchange calendar, zero-volume
days, adjustment sanity around known splits/bonuses) against the Phase-2 NSE
basket and reports a GO / NO-GO. This module is an AUDIT ONLY: it fetches and
analyzes, and writes nothing to the raw/clean/features/forecasts pipeline. NSE
integration happens only after the audit is reviewed and approved.

Adjustment note (verified 2026-07-13): yfinance's "Close" is already
split/bonus-adjusted (only dividends separate it from "Adj Close"), so a
correctly-served series has NO split-sized raw jumps and its close/adj_close
ratio converges to 1.0 at the present. A split-sized raw jump would therefore
signal a MISSED corporate action — the adjustment red flag this audit looks for.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from volrisk.providers.base import OHLCVProvider
from volrisk.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

NSE_CALENDAR = "XNSE"

#: Phase-2 basket. ^NSEI is the index (reference data, like ^VIX in the US
#: basket); the other four are the modelable equities.
NSE_INDEX = "^NSEI"
NSE_EQUITIES = ("RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS")
NSE_TICKERS = (NSE_INDEX, *NSE_EQUITIES)

#: Known corporate actions, for cross-referencing the adjustment check.
KNOWN_CORPORATE_ACTIONS: dict[str, list[tuple[date, str]]] = {
    "RELIANCE.NS": [(date(2017, 9, 7), "1:1 bonus")],
    "TCS.NS": [(date(2018, 6, 2), "1:1 bonus")],
    "INFY.NS": [(date(2018, 9, 4), "1:1 bonus")],
    "HDFCBANK.NS": [(date(2019, 9, 19), "face-value split 2:1")],
}

#: A single-day raw move above this is not a market move — it is an unadjusted
#: split/bonus. Zero such jumps => every corporate action was back-adjusted.
SPLIT_JUMP_THRESHOLD = 0.35

#: GO thresholds.
MAX_GAP_RATE = 0.010  # missing calendar sessions / expected
MAX_ZERO_VOLUME_RATE = 0.010  # zero-volume bars / bars (equities only)
MAX_LAST_RATIO_DEVIATION = 0.01  # |close/adj_close - 1| on the most recent bar


@dataclass
class TickerAudit:
    ticker: str
    is_index: bool
    n_bars: int
    first_bar: date
    last_bar: date
    expected_sessions: int
    missing_sessions: list[date]  # calendar open, no bar
    special_sessions: list[date]  # bar exists, calendar says closed
    zero_volume_days: list[date]
    last_close_adj_ratio: float
    max_raw_move: float
    max_adj_move: float
    split_sized_jumps: list[date]
    corporate_actions_ok: bool
    notes: list[str] = field(default_factory=list)

    @property
    def gap_rate(self) -> float:
        return len(self.missing_sessions) / self.expected_sessions

    @property
    def zero_volume_rate(self) -> float:
        return len(self.zero_volume_days) / self.n_bars

    @property
    def adjustment_ok(self) -> bool:
        return (
            abs(self.last_close_adj_ratio - 1.0) < MAX_LAST_RATIO_DEVIATION
            and not self.split_sized_jumps
        )

    def passes(self) -> bool:
        """GO for this ticker. The index is exempt from the equity volume check."""
        gaps_ok = self.gap_rate < MAX_GAP_RATE
        volume_ok = self.is_index or self.zero_volume_rate < MAX_ZERO_VOLUME_RATE
        return gaps_ok and volume_ok and self.adjustment_ok


def audit_ticker(df: pd.DataFrame, calendar, ticker: str) -> TickerAudit:
    """Run the three data-quality checks on one ticker's canonical bars."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    first, last = df["trade_date"].iloc[0], df["trade_date"].iloc[-1]
    sessions = {ts.date() for ts in calendar.schedule(start_date=first, end_date=last).index}
    present = set(df["trade_date"])

    raw_ret = np.log(df["close"] / df["close"].shift(1))
    adj_ret = np.log(df["adj_close"] / df["adj_close"].shift(1))
    split_jumps = df.loc[raw_ret.abs() > SPLIT_JUMP_THRESHOLD, "trade_date"].tolist()

    audit = TickerAudit(
        ticker=ticker,
        is_index=ticker == NSE_INDEX,
        n_bars=len(df),
        first_bar=first,
        last_bar=last,
        expected_sessions=len(sessions),
        missing_sessions=sorted(sessions - present),
        special_sessions=sorted(present - sessions),
        zero_volume_days=df.loc[df["volume"].fillna(0) == 0, "trade_date"].tolist(),
        last_close_adj_ratio=float(df["close"].iloc[-1] / df["adj_close"].iloc[-1]),
        max_raw_move=float(raw_ret.abs().max()),
        max_adj_move=float(adj_ret.abs().max()),
        split_sized_jumps=split_jumps,
        corporate_actions_ok=len(split_jumps) == 0,
    )
    for ex_date, desc in KNOWN_CORPORATE_ACTIONS.get(ticker, []):
        lo, hi = ex_date - pd.Timedelta(days=7), ex_date + pd.Timedelta(days=7)
        near = df[(df["trade_date"] >= lo) & (df["trade_date"] <= hi)]
        adj_jump = float(np.log(near["adj_close"]).diff().abs().max()) if not near.empty else np.nan
        smooth = near.empty is False and adj_jump < SPLIT_JUMP_THRESHOLD
        verdict = "continuous" if smooth else "DISCONTINUOUS — check"
        audit.notes.append(f"{desc} (~{ex_date}): adjusted series {verdict}")
    return audit


def run_audit(
    provider: OHLCVProvider, years: int = 10, tickers: tuple[str, ...] = NSE_TICKERS
) -> list[TickerAudit]:
    """Fetch and audit every NSE ticker. Network happens here; analysis is pure."""
    calendar = mcal.get_calendar(NSE_CALENDAR)
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=years)).date()
    audits = []
    for ticker in tickers:
        df = provider.fetch_daily_ohlcv(ticker, start, end)
        audits.append(audit_ticker(df, calendar, ticker))
    return audits


def _format_report(audits: list[TickerAudit]) -> str:
    lines = [
        f"NSE data-quality audit ({NSE_CALENDAR} calendar). "
        "Adjustment red flag = split-sized raw jump (missed corporate action).",
        "",
        f"{'ticker':>12} {'bars':>5} {'exp':>5} {'miss':>5} {'gap%':>6} "
        f"{'spec':>5} {'zerovol':>8} {'c/adj':>7} {'maxraw':>7} {'splitJmp':>8} {'verdict':>8}",
    ]
    for a in audits:
        zv = f"{len(a.zero_volume_days)}/{a.zero_volume_rate * 100:.1f}%"
        lines.append(
            f"{a.ticker:>12} {a.n_bars:>5} {a.expected_sessions:>5} {len(a.missing_sessions):>5} "
            f"{a.gap_rate * 100:>5.2f}% {len(a.special_sessions):>5} {zv:>8} "
            f"{a.last_close_adj_ratio:>7.4f} {a.max_raw_move * 100:>6.1f}% "
            f"{len(a.split_sized_jumps):>8} {'GO' if a.passes() else 'NO-GO':>8}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE (India) data-quality audit — Step 10 gate.")
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    audits = run_audit(YFinanceProvider(), years=args.years)
    print(_format_report(audits))

    print("\nAdjustment cross-checks against known corporate actions:")
    for a in audits:
        for note in a.notes:
            print(f"  {a.ticker}: {note}")

    print("\nCalendar-anomaly detail (surplus = NSE special sessions the calendar omits):")
    for a in audits:
        if a.special_sessions:
            shown = ", ".join(str(d) for d in a.special_sessions[:12])
            print(f"  {a.ticker} special sessions ({len(a.special_sessions)}): {shown}")
        if a.missing_sessions:
            shown = ", ".join(str(d) for d in a.missing_sessions[:12])
            print(f"  {a.ticker} missing sessions ({len(a.missing_sessions)}): {shown}")
        if a.ticker == NSE_INDEX:  # equity anomaly sets are identical; show once
            break

    n_go = sum(a.passes() for a in audits)
    print(f"\nOverall: {n_go}/{len(audits)} tickers pass.")


if __name__ == "__main__":
    main()
