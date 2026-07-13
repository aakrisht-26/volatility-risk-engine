"""Offline tests for the NSE audit logic. Fixture-built frames — no network.

Verifies each check fires: a missing calendar session, a special (non-session)
bar, a zero-volume day, and — the load-bearing one — an unadjusted split showing
up as a split-sized raw jump (a missed corporate action).
"""

from datetime import date

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from volrisk.audit.nse import (
    SPLIT_JUMP_THRESHOLD,
    audit_ticker,
)

XNSE = mcal.get_calendar("XNSE")


def clean_frame(ticker: str = "RELIANCE.NS", n: int = 60) -> pd.DataFrame:
    """A well-behaved canonical frame on real XNSE sessions, ratio -> 1.0 today."""
    dates = [ts.date() for ts in XNSE.schedule("2024-01-01", "2024-12-31").index[:n]]
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], np.full(n - 1, 0.001)])))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,  # ratio 1.0, converges cleanly
            "volume": pd.array([1_000_000] * n, dtype="Int64"),
        }
    )


def test_clean_frame_passes_every_check() -> None:
    audit = audit_ticker(clean_frame(), XNSE, "RELIANCE.NS")

    assert audit.missing_sessions == []
    assert audit.special_sessions == []
    assert audit.zero_volume_days == []
    assert audit.split_sized_jumps == []
    assert audit.adjustment_ok
    assert audit.passes()


def test_missing_session_is_detected() -> None:
    df = clean_frame(n=60)
    dropped = df["trade_date"].iloc[30]
    df = df[df["trade_date"] != dropped].reset_index(drop=True)

    audit = audit_ticker(df, XNSE, "RELIANCE.NS")

    assert dropped in audit.missing_sessions
    assert audit.gap_rate > 0


def test_special_non_session_bar_is_detected() -> None:
    df = clean_frame(n=40)
    # 2024-01-26 (Republic Day) is an NSE holiday: a bar there is a special session.
    holiday = date(2024, 1, 26)
    extra = df.iloc[[0]].copy()
    extra["trade_date"] = holiday
    df = pd.concat([df, extra], ignore_index=True)

    audit = audit_ticker(df, XNSE, "RELIANCE.NS")

    assert holiday in audit.special_sessions


def test_zero_volume_day_is_flagged() -> None:
    df = clean_frame(n=40)
    df.loc[10, "volume"] = 0

    audit = audit_ticker(df, XNSE, "RELIANCE.NS")

    assert df["trade_date"].iloc[10] in audit.zero_volume_days
    assert audit.zero_volume_rate > 0


def test_unadjusted_split_shows_as_split_sized_jump() -> None:
    df = clean_frame(n=40)
    # Halve the raw close from row 20 on (an un-back-adjusted 1:1 bonus) while
    # keeping adj_close continuous — this is the "missed corporate action" case.
    df.loc[20:, "close"] = df.loc[20:, "close"] / 2.0

    audit = audit_ticker(df, XNSE, "RELIANCE.NS")

    assert audit.split_sized_jumps  # a split-sized raw jump was caught
    assert audit.max_raw_move > SPLIT_JUMP_THRESHOLD
    assert not audit.corporate_actions_ok
    assert not audit.adjustment_ok
    assert not audit.passes()


def test_last_ratio_far_from_one_fails_adjustment() -> None:
    df = clean_frame(n=40)
    df["adj_close"] = df["close"] * 0.80  # present-day ratio stuck at 1.25

    audit = audit_ticker(df, XNSE, "RELIANCE.NS")

    assert not audit.adjustment_ok


def test_index_is_exempt_from_volume_threshold() -> None:
    df = clean_frame("^NSEI", n=40)
    df["volume"] = pd.array([0] * 40, dtype="Int64")  # index volume all zero

    audit = audit_ticker(df, XNSE, "^NSEI")

    assert audit.is_index
    assert audit.passes()  # zero-volume does not fail the index
