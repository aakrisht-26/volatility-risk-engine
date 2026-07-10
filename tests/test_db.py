"""Database-layer tests.

The statement-compilation and record-conversion tests run everywhere. The
integration tests need a disposable Postgres database and are skipped unless
``VOLRISK_TEST_DATABASE_URL`` is set — locally via ``uv run --env-file .env
pytest``, in CI via a postgres:16 service container. They never touch the app
database.
"""

import os
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from volrisk.db.loaders import (
    build_upsert_statement,
    frame_to_records,
    raw_daily_bars_count,
    upsert_daily_bars,
)

TEST_DB_URL = os.environ.get("VOLRISK_TEST_DATABASE_URL")

requires_db = pytest.mark.skipif(not TEST_DB_URL, reason="VOLRISK_TEST_DATABASE_URL not set")


def canonical_frame(ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [187.15, 184.22],
            "high": [188.44, 185.88],
            "low": [183.885, 183.43],
            "close": [185.64, 184.25],
            "adj_close": [184.55, 183.17],
            "volume": pd.array([82488700, 58414500], dtype="Int64"),
        }
    )


# --- offline ---


def test_upsert_statement_targets_natural_key() -> None:
    sql = str(build_upsert_statement().compile(dialect=postgresql.dialect()))

    assert "INSERT INTO raw.daily_bars" in sql
    assert "ON CONFLICT (ticker, trade_date) DO UPDATE" in sql
    assert "excluded.close" in sql  # revisable bars: conflicting rows are updated, not skipped


def test_frame_to_records_converts_na_to_none() -> None:
    df = canonical_frame()
    df["volume"] = pd.array([82488700, None], dtype="Int64")

    records = frame_to_records(df)

    assert records[0]["volume"] == 82488700
    assert records[1]["volume"] is None


# --- integration (disposable test database only) ---


@pytest.fixture
def clean_table(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE raw.daily_bars"))
    return db_engine


@requires_db
def test_migrations_rerun_is_idempotent(db_engine) -> None:
    from volrisk.db.migrate import apply_migrations

    assert apply_migrations(db_engine) == []  # everything already applied by the fixture


@requires_db
def test_upsert_inserts_rows(clean_table) -> None:
    n = upsert_daily_bars(clean_table, canonical_frame(), context="test")

    assert n == 2
    assert raw_daily_bars_count(clean_table) == 2


@requires_db
def test_upsert_rerun_adds_zero_rows(clean_table) -> None:
    upsert_daily_bars(clean_table, canonical_frame(), context="test")
    before = raw_daily_bars_count(clean_table)

    upsert_daily_bars(clean_table, canonical_frame(), context="test")

    assert before == 2
    assert raw_daily_bars_count(clean_table) == 2


@requires_db
def test_upsert_revises_changed_bar_in_place(clean_table) -> None:
    df = canonical_frame()
    upsert_daily_bars(clean_table, df, context="test")

    revised = df.copy()
    revised.loc[0, "close"] = 186.10  # same-day partial bar, finalized after the close
    upsert_daily_bars(clean_table, revised, context="test")

    assert raw_daily_bars_count(clean_table) == 2  # revised, not duplicated
    with clean_table.connect() as conn:
        stored = conn.execute(
            text("SELECT close FROM raw.daily_bars WHERE ticker = 'AAPL' AND trade_date = :d"),
            {"d": date(2024, 1, 2)},
        ).scalar_one()
    assert stored == 186.10
