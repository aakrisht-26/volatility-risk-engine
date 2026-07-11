"""Cross-check the SQL feature layer against an independent pandas recomputation.

Step-6 acceptance: for a ticker, every feature column computed by the SQL
window functions must match a from-scratch pandas recomputation within
tolerance, with identical NULL masks. This cross-check IS the test — it also
runs in CI against synthetic data (see tests/test_features.py).

Usage::

    uv run python -m volrisk.features.crosscheck [--ticker AAPL] [--tolerance 1e-12]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = (
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "r2",
    "park_var",
    "gk_var",
    "rvol_5",
    "rvol_21",
    "rvol_63",
    "har_rv_w",
    "har_rv_m",
    "target_gk_var_next",
)


def recompute_features_pandas(clean: pd.DataFrame) -> pd.DataFrame:
    """Recompute every feature column from a ticker's clean bars, mirroring the SQL.

    Deliberately written against pandas primitives (shift/rolling) rather than
    any shared helper, so it is an independent implementation of the same
    definitions.
    """
    df = clean.sort_values("trade_date").reset_index(drop=True).copy()
    r = df["log_return"]
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])

    df["r2"] = r * r
    df["park_var"] = hl * hl / (4 * np.log(2))
    df["gk_var"] = 0.5 * hl * hl - (2 * np.log(2) - 1) * co * co
    for k in range(1, 6):
        df[f"ret_lag_{k}"] = r.shift(k)
    for window, col in ((5, "rvol_5"), (21, "rvol_21"), (63, "rvol_63")):
        df[col] = np.sqrt(252 * df["r2"].rolling(window, min_periods=window).mean())
    df["har_rv_w"] = df["gk_var"].rolling(5, min_periods=5).mean()
    df["har_rv_m"] = df["gk_var"].rolling(22, min_periods=22).mean()
    df["target_gk_var_next"] = df["gk_var"].shift(-1)
    return df


def compare_features(
    sql_df: pd.DataFrame,
    pandas_df: pd.DataFrame,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, float]:
    """Max |diff| per column; raises if row alignment or NULL masks disagree."""
    if len(sql_df) != len(pandas_df):
        raise AssertionError(f"row counts differ: {len(sql_df)} vs {len(pandas_df)}")
    if list(sql_df["trade_date"]) != list(pandas_df["trade_date"]):
        raise AssertionError("trade_date alignment mismatch between SQL and pandas frames")

    diffs: dict[str, float] = {}
    for col in columns:
        a = sql_df[col].to_numpy(dtype=float)
        b = pandas_df[col].to_numpy(dtype=float)
        if not (np.isnan(a) == np.isnan(b)).all():
            raise AssertionError(f"{col}: NULL masks differ between SQL and pandas")
        mask = ~np.isnan(a)
        diffs[col] = float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
    return diffs


def crosscheck_ticker(engine: Engine, ticker: str) -> dict[str, float]:
    """Compare SQL features vs pandas recomputation for one ticker."""
    clean = pd.read_sql_query(
        text(
            "SELECT ticker, trade_date, open, high, low, close, adj_close, volume, log_return"
            " FROM clean.daily_bars WHERE ticker = :t ORDER BY trade_date"
        ),
        engine,
        params={"t": ticker},
    )
    if clean.empty:
        raise SystemExit(f"no clean bars for ticker {ticker!r}")
    feats = pd.read_sql_query(
        text("SELECT * FROM features.daily_features WHERE ticker = :t ORDER BY trade_date"),
        engine,
        params={"t": ticker},
    )
    return compare_features(feats, recompute_features_pandas(clean))


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL-vs-pandas feature cross-check.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    diffs = crosscheck_ticker(get_engine(), args.ticker)

    print(f"\n=== Feature cross-check: SQL vs pandas, {args.ticker} ===")
    failures = 0
    for col, diff in diffs.items():
        verdict = "OK" if diff < args.tolerance else "FAIL"
        failures += verdict == "FAIL"
        print(f"  {col:>20}  max|diff| = {diff:.3e}  {verdict}")
    if failures:
        raise SystemExit(f"{failures} column(s) exceeded tolerance {args.tolerance}")
    print(f"all {len(diffs)} columns within {args.tolerance}")


if __name__ == "__main__":
    main()
