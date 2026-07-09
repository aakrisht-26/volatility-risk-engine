"""Idempotent loaders: raw parquet landing zone -> Postgres ``raw`` schema.

Upserts on the natural key (ticker, trade_date). ON CONFLICT DO UPDATE rather
than DO NOTHING because recent bars are revisable by design: a fetch during US
market hours lands today's in-progress bar, and the Step-11 nightly job
re-fetches a trailing window (see CLAUDE.md data decisions). Re-running any
load therefore adds zero duplicate rows while still healing revised bars.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Double,
    Engine,
    MetaData,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import insert

from volrisk.validate.schemas import validate_daily_bars

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

_UPDATABLE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


def build_upsert_statement():
    """INSERT ... ON CONFLICT (ticker, trade_date) DO UPDATE for daily bars."""
    stmt = insert(DAILY_BARS)
    return stmt.on_conflict_do_update(
        index_elements=["ticker", "trade_date"],
        set_={col: stmt.excluded[col] for col in _UPDATABLE_COLUMNS} | {"loaded_at": func.now()},
    )


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


def raw_daily_bars_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM raw.daily_bars")).scalar_one()
