"""Tests for the SQL feature layer.

The heart of Step 6's acceptance is the SQL-vs-pandas cross-check: a synthetic
but realistic OHLC series is seeded into the disposable test database, built
into features by the real SQL, and compared column-by-column against an
independent pandas recomputation. Offline tests pin the structural semantics
(lag shifts, full-window NULL warm-ups).
"""

import os

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from sqlalchemy import text

from volrisk.db.loaders import upsert_clean_daily_bars
from volrisk.features.build import build_features
from volrisk.features.crosscheck import compare_features, recompute_features_pandas
from volrisk.transform.returns import add_log_returns

requires_db = pytest.mark.skipif(
    not os.environ.get("VOLRISK_TEST_DATABASE_URL"), reason="VOLRISK_TEST_DATABASE_URL not set"
)


def synthetic_clean_frame(ticker: str = "SYN", n: int = 80, seed: int = 7) -> pd.DataFrame:
    """Deterministic, constraint-satisfying OHLC series on real XNYS sessions."""
    rng = np.random.default_rng(seed)
    sessions = mcal.get_calendar("XNYS").schedule("2024-01-02", "2024-12-31").index[:n]
    rets = rng.normal(0.0, 0.02, n)
    rets[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0.0, 0.003, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.004, n)))

    df = pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": [ts.date() for ts in sessions],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": pd.array([1_000_000] * n, dtype="Int64"),
        }
    )
    return add_log_returns(df)


# --- offline: structural semantics of the recomputation ---


def test_lagged_returns_shift_within_ticker() -> None:
    df = recompute_features_pandas(synthetic_clean_frame(n=10))

    assert df["ret_lag_1"].iloc[5] == df["log_return"].iloc[4]
    assert df["ret_lag_5"].iloc[9] == df["log_return"].iloc[4]
    assert pd.isna(df["ret_lag_5"].iloc[4])


def test_rolling_stats_are_null_until_window_is_full() -> None:
    df = recompute_features_pandas(synthetic_clean_frame(n=70))

    # log_return (hence r2) is NULL on row 0, so the first full 5-observation
    # window of r2 ends on row 5, the 21-window on row 21, the 63-window on row 63.
    assert df["rvol_5"].iloc[:5].isna().all() and pd.notna(df["rvol_5"].iloc[5])
    assert df["rvol_21"].iloc[:21].isna().all() and pd.notna(df["rvol_21"].iloc[21])
    assert df["rvol_63"].iloc[:63].isna().all() and pd.notna(df["rvol_63"].iloc[63])
    # gk_var has no warm-up NULL, so HAR windows fill one row earlier.
    assert df["har_rv_w"].iloc[:4].isna().all() and pd.notna(df["har_rv_w"].iloc[4])
    assert df["har_rv_m"].iloc[:21].isna().all() and pd.notna(df["har_rv_m"].iloc[21])


def test_parkinson_variance_is_nonnegative_and_target_is_next_day() -> None:
    df = recompute_features_pandas(synthetic_clean_frame(n=30))

    assert (df["park_var"] >= 0).all()
    assert df["target_gk_var_next"].iloc[10] == df["gk_var"].iloc[11]
    assert pd.isna(df["target_gk_var_next"].iloc[-1])


# --- integration: the cross-check IS the test ---


@requires_db
def test_sql_features_match_pandas_recomputation(db_engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE clean.daily_bars"))
        conn.execute(text("TRUNCATE features.daily_features"))
    df = synthetic_clean_frame()
    upsert_clean_daily_bars(db_engine, df, context="synthetic")

    n_first = build_features(db_engine)
    n_second = build_features(db_engine)  # idempotent re-run

    assert n_first == n_second == len(df)
    feats = pd.read_sql_query(
        text("SELECT * FROM features.daily_features WHERE ticker = 'SYN' ORDER BY trade_date"),
        db_engine,
    )
    diffs = compare_features(feats, recompute_features_pandas(df))
    assert max(diffs.values()) < 1e-12, diffs


@requires_db
def test_feature_windows_do_not_leak_across_tickers(db_engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE clean.daily_bars"))
        conn.execute(text("TRUNCATE features.daily_features"))
    upsert_clean_daily_bars(db_engine, synthetic_clean_frame("AAA", n=30, seed=1), context="AAA")
    upsert_clean_daily_bars(db_engine, synthetic_clean_frame("BBB", n=30, seed=2), context="BBB")

    build_features(db_engine)

    feats = pd.read_sql_query(
        text("SELECT ticker, ret_lag_1 FROM features.daily_features ORDER BY ticker, trade_date"),
        db_engine,
    )
    # Each ticker's first row must have a NULL lag — a non-NULL value there
    # would mean the window read the other ticker's partition.
    first_rows = feats.groupby("ticker").head(1)
    assert first_rows["ret_lag_1"].isna().all()
    assert len(feats) == 60
