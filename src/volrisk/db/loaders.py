"""Idempotent loaders: raw parquet landing zone -> Postgres ``raw`` schema.

Upserts on the natural key (ticker, trade_date). ON CONFLICT DO UPDATE rather
than DO NOTHING because recent bars are revisable by design: a fetch during US
market hours lands today's in-progress bar, and the Step-11 nightly job
re-fetches a trailing window (see CLAUDE.md data decisions). Re-running any
load therefore adds zero duplicate rows while still healing revised bars.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Double,
    Engine,
    Integer,
    MetaData,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import insert

from volrisk.validate.schemas import (
    validate_clean_daily_bars,
    validate_daily_bars,
    validate_variance_forecasts,
)

logger = logging.getLogger(__name__)

# Static mirror of db/migrations/001_raw.sql — no runtime reflection needed.
DAILY_BARS = Table(
    "daily_bars",
    MetaData(schema="raw"),
    Column("ticker", Text, primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("open", Double, nullable=False),
    Column("high", Double, nullable=False),
    Column("low", Double, nullable=False),
    Column("close", Double, nullable=False),
    Column("adj_close", Double, nullable=False),
    Column("volume", BigInteger),
    Column("loaded_at", DateTime(timezone=True), server_default=func.now()),
)

# Same natural-key layout as raw, plus the computed log return.
CLEAN_DAILY_BARS = Table(
    "daily_bars",
    MetaData(schema="clean"),
    Column("ticker", Text, primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("open", Double, nullable=False),
    Column("high", Double, nullable=False),
    Column("low", Double, nullable=False),
    Column("close", Double, nullable=False),
    Column("adj_close", Double, nullable=False),
    Column("volume", BigInteger),
    Column("log_return", Double),
    Column("loaded_at", DateTime(timezone=True), server_default=func.now()),
)

# Next-day variance forecasts; var_forecast is daily variance in RETURN units
# (see db/migrations/004_forecasts.sql — the units comment there is load-bearing).
VARIANCE_FORECASTS = Table(
    "daily_variance",
    MetaData(schema="forecasts"),
    Column("ticker", Text, primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("var_forecast", Double, nullable=False),
    Column("loaded_at", DateTime(timezone=True), server_default=func.now()),
)

_UPDATABLE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
_CLEAN_UPDATABLE_COLUMNS = (*_UPDATABLE_COLUMNS, "log_return")


def build_upsert(table: Table, conflict_cols: Sequence[str], update_cols: Sequence[str]):
    """INSERT ... ON CONFLICT (natural key) DO UPDATE — revisable rows, zero duplicates.

    The table's timestamp column (loaded_at or computed_at) is refreshed on
    every conflict so a row always records its latest write.
    """
    stmt = insert(table)
    set_ = {col: stmt.excluded[col] for col in update_cols}
    for ts_col in ("loaded_at", "computed_at"):
        if ts_col in table.c:
            set_[ts_col] = func.now()
    return stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=set_)


def build_upsert_statement():
    """INSERT ... ON CONFLICT (ticker, trade_date) DO UPDATE for raw daily bars."""
    return build_upsert(DAILY_BARS, ("ticker", "trade_date"), _UPDATABLE_COLUMNS)


def read_landing_parquet(path: Path) -> pd.DataFrame:
    """Read one landing-zone parquet back into the canonical frame shape."""
    df = pd.read_parquet(path)
    # Parquet readers may hand date32 back as datetime64; the canonical
    # contract (and pandera) require plain python dates.
    if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df["trade_date"] = df["trade_date"].map(lambda ts: ts.date())
    return df


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    """Canonical frame -> driver records (pandas NA -> None)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def upsert_daily_bars(engine: Engine, df: pd.DataFrame, context: str = "") -> int:
    """Validate one canonical batch and upsert it; returns the batch row count."""
    validate_daily_bars(df, context=context)
    records = frame_to_records(df)
    with engine.begin() as conn:
        conn.execute(build_upsert_statement(), records)
    return len(records)


def upsert_clean_daily_bars(engine: Engine, df: pd.DataFrame, context: str = "") -> int:
    """Validate one clean batch (canonical + log_return) and upsert it."""
    validate_clean_daily_bars(df, context=context)
    records = frame_to_records(df)
    with engine.begin() as conn:
        conn.execute(
            build_upsert(CLEAN_DAILY_BARS, ("ticker", "trade_date"), _CLEAN_UPDATABLE_COLUMNS),
            records,
        )
    return len(records)


# Derived evaluation metrics (Step 8). No pandera layer here: inputs are the
# already-validated forecasts and features tables, and the values are computed,
# not ingested.
ABLATION_METRICS = Table(
    "ablation_metrics",
    MetaData(schema="forecasts"),
    Column("ticker", Text, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("n_obs", Integer, nullable=False),
    Column("qlike", Double, nullable=False),
    Column("rmse_ann_vol_pct", Double, nullable=False),
    Column("eval_start", Date, nullable=False),
    Column("eval_end", Date, nullable=False),
    Column("computed_at", DateTime(timezone=True), server_default=func.now()),
)

_ABLATION_UPDATABLE = ("n_obs", "qlike", "rmse_ann_vol_pct", "eval_start", "eval_end")


def upsert_ablation_metrics(engine: Engine, df: pd.DataFrame) -> int:
    """Upsert computed ablation metrics keyed by (ticker, model)."""
    records = df.to_dict("records")
    stmt = build_upsert(ABLATION_METRICS, ("ticker", "model"), _ABLATION_UPDATABLE)
    with engine.begin() as conn:
        conn.execute(stmt, records)
    return len(records)


def upsert_variance_forecasts(engine: Engine, df: pd.DataFrame, context: str = "") -> int:
    """Validate one batch of variance forecasts and upsert it."""
    validate_variance_forecasts(df, context=context)
    records = frame_to_records(df)
    with engine.begin() as conn:
        conn.execute(
            build_upsert(VARIANCE_FORECASTS, ("ticker", "trade_date", "model"), ("var_forecast",)),
            records,
        )
    return len(records)


# VaR coverage backtest tables (Step 9). Computed from validated forecasts and
# clean returns; the backtest recomputes every row each run, so storage is a
# truncate-and-insert (full replace) rather than an upsert — this correctly
# drops stale rows (e.g. _cal variants) when a later run omits them.
VAR_COVERAGE = Table(
    "var_coverage",
    MetaData(schema="forecasts"),
    Column("ticker", Text, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("level", Integer, primary_key=True),
    Column("n_obs", Integer, nullable=False),
    Column("expected_breaches", Double, nullable=False),
    Column("observed_breaches", Integer, nullable=False),
    Column("breach_rate", Double, nullable=False),
    Column("kupiec_lr", Double, nullable=False),
    Column("kupiec_p", Double, nullable=False),
    Column("eval_start", Date, nullable=False),
    Column("eval_end", Date, nullable=False),
    Column("computed_at", DateTime(timezone=True), server_default=func.now()),
)

VAR_BREACHES = Table(
    "var_breaches",
    MetaData(schema="forecasts"),
    Column("ticker", Text, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("level", Integer, primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("log_return", Double, nullable=False),
    Column("var_threshold", Double, nullable=False),
)

_VAR_COVERAGE_COLUMNS = (
    "ticker",
    "model",
    "level",
    "n_obs",
    "expected_breaches",
    "observed_breaches",
    "breach_rate",
    "kupiec_lr",
    "kupiec_p",
    "eval_start",
    "eval_end",
)
_VAR_BREACH_COLUMNS = ("ticker", "model", "level", "trade_date", "log_return", "var_threshold")


def store_var_results(
    engine: Engine, coverage: pd.DataFrame, breaches: pd.DataFrame
) -> tuple[int, int]:
    """Replace all VaR coverage + breach rows in one transaction (idempotent)."""
    cov_records = coverage[list(_VAR_COVERAGE_COLUMNS)].to_dict("records")
    br_records = (
        breaches[list(_VAR_BREACH_COLUMNS)].to_dict("records") if not breaches.empty else []
    )
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE forecasts.var_coverage, forecasts.var_breaches"))
        if cov_records:
            conn.execute(VAR_COVERAGE.insert(), cov_records)
        if br_records:
            conn.execute(VAR_BREACHES.insert(), br_records)
    return len(cov_records), len(br_records)


def raw_daily_bars_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM raw.daily_bars")).scalar_one()
