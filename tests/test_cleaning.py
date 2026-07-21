"""Tests for calendar alignment, the partial-bar policy, and log returns.

Offline: pandas-market-calendars ships exchange rules as code, no network.
Scenario dates are early-January 2024 XNYS sessions (01-02..01-05, 01-08,
01-09); 2024-01-06 is a Saturday. The XNYS close is 21:00 UTC in January, so
an as-of of 18:00 UTC is "during market hours" and 22:00 UTC is "after close".
"""

import os
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from sqlalchemy import text

from volrisk.db.loaders import upsert_clean_daily_bars
from volrisk.transform.cleaning import clean_ticker_frame, last_completed_session, next_session
from volrisk.transform.returns import add_log_returns, telescoping_check

XNYS = mcal.get_calendar("XNYS")

requires_db = pytest.mark.skipif(
    not os.environ.get("VOLRISK_TEST_DATABASE_URL"), reason="VOLRISK_TEST_DATABASE_URL not set"
)

DURING_HOURS = datetime(2024, 1, 9, 18, 0, tzinfo=UTC)
AFTER_CLOSE = datetime(2024, 1, 9, 22, 0, tzinfo=UTC)


def make_raw_frame(dates: list[date], ticker: str = "TEST") -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "adj_close": [100.5 + i for i in range(n)],
            "volume": pd.array([1_000_000] * n, dtype="Int64"),
        }
    )


def test_last_completed_session_flips_at_market_close() -> None:
    assert last_completed_session(XNYS, DURING_HOURS) == date(2024, 1, 8)
    assert last_completed_session(XNYS, AFTER_CLOSE) == date(2024, 1, 9)


def test_next_session_skips_weekends_and_holidays() -> None:
    assert next_session(date(2024, 1, 3)) == date(2024, 1, 4)  # plain weekday
    # Friday 2024-01-12 -> MLK Monday is a holiday -> Tuesday 2024-01-16.
    assert next_session(date(2024, 1, 12)) == date(2024, 1, 16)


def test_clean_frame_classifies_missing_extra_and_partial() -> None:
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        # 2024-01-04 session deliberately missing
        date(2024, 1, 5),
        date(2024, 1, 6),  # Saturday: non-session row
        date(2024, 1, 8),
        date(2024, 1, 9),  # in-progress session at DURING_HOURS: partial
    ]

    clean, report = clean_ticker_frame(make_raw_frame(dates), XNYS, DURING_HOURS)

    assert report.missing_sessions == [date(2024, 1, 4)]
    assert report.non_session_rows == [date(2024, 1, 6)]
    assert report.partial_rows == [date(2024, 1, 9)]
    assert report.expected_sessions == 5  # 01-02, 01-03, 01-04, 01-05, 01-08
    assert report.bars_present == 6
    assert report.clean_rows == 4
    assert list(clean["trade_date"]) == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    assert pd.isna(clean["log_return"].iloc[0])
    assert clean["log_return"].iloc[1:].notna().all()


def test_partial_bar_enters_clean_after_session_completes() -> None:
    dates = [date(2024, 1, 8), date(2024, 1, 9)]

    clean, report = clean_ticker_frame(make_raw_frame(dates), XNYS, AFTER_CLOSE)

    assert report.partial_rows == []
    assert list(clean["trade_date"]) == dates


def test_log_returns_first_row_null_then_ratio() -> None:
    df = add_log_returns(make_raw_frame([date(2024, 1, 2), date(2024, 1, 3)]))

    assert pd.isna(df["log_return"].iloc[0])
    expected = np.log(df["adj_close"].iloc[1] / df["adj_close"].iloc[0])
    assert df["log_return"].iloc[1] == pytest.approx(expected, rel=1e-15)


def test_telescoping_identity_holds() -> None:
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    df = add_log_returns(make_raw_frame(dates))

    total, endpoints = telescoping_check(df)

    assert total == pytest.approx(endpoints, abs=1e-12)


@requires_db
def test_clean_upsert_rerun_adds_zero_rows(db_engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE clean.daily_bars"))
    df = add_log_returns(make_raw_frame([date(2024, 1, 2), date(2024, 1, 3)]))

    upsert_clean_daily_bars(db_engine, df, context="test")
    upsert_clean_daily_bars(db_engine, df, context="test")

    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM clean.daily_bars")).scalar_one()
    assert n == 2
