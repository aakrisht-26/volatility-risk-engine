"""Tests for the backfill writer, using an in-memory fake provider. No network.

The monotonic landing-zone guard is the Step-11 addition: with an anchored
inception a legitimate refresh can only grow or match a ticker's history, so a
shrinking fetch must be refused (and --force must override after human
investigation).
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from volrisk.ingest.backfill import backfill_ticker, parquet_path, run_backfill
from volrisk.providers.base import OHLCVProvider


def canonical_frame(ticker: str = "AAPL", n: int = 2) -> pd.DataFrame:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)][:n]
    base = {
        "open": [187.15, 184.22, 183.10, 182.40],
        "high": [188.44, 185.88, 184.20, 183.55],
        "low": [183.885, 183.43, 182.05, 181.20],
        "close": [185.64, 184.25, 183.20, 182.10],
        "adj_close": [184.55, 183.17, 182.13, 181.04],
    }
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": dates,
            **{col: values[:n] for col, values in base.items()},
            "volume": pd.array([82488700, 58414500, 61000000, 59500000][:n], dtype="Int64"),
        }
    )


class FakeProvider(OHLCVProvider):
    """Serves a fixed canonical frame; lets backfill be tested without network."""

    def __init__(self, n: int = 2) -> None:
        self.n = n

    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return canonical_frame(ticker, n=self.n)


def test_parquet_path_strips_index_caret(tmp_path: Path) -> None:
    assert parquet_path(tmp_path, "^GSPC") == tmp_path / "GSPC.parquet"
    assert parquet_path(tmp_path, "AAPL") == tmp_path / "AAPL.parquet"


def test_backfill_ticker_writes_parquet(tmp_path: Path) -> None:
    result = backfill_ticker(FakeProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    assert result.rows == 2
    assert result.written and not result.guarded
    written = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert len(written) == 2
    assert not list(tmp_path.glob("*.tmp"))  # atomic write left no temp files


def test_backfill_rerun_is_idempotent(tmp_path: Path) -> None:
    backfill_ticker(FakeProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)
    second = backfill_ticker(FakeProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    assert second.written  # equal row count passes the guard
    written = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert len(written) == 2  # file replaced wholesale — zero duplicate rows
    assert written["trade_date"].is_unique


def test_run_backfill_covers_all_tickers(tmp_path: Path) -> None:
    results = run_backfill(
        FakeProvider(), ("^GSPC", "AAPL"), start=date(2024, 1, 1), out_dir=tmp_path
    )

    assert {r.ticker: r.rows for r in results} == {"^GSPC": 2, "AAPL": 2}
    assert all(r.written for r in results)
    assert (tmp_path / "GSPC.parquet").exists()
    assert (tmp_path / "AAPL.parquet").exists()


# --- monotonic landing-zone guard ---


def test_guard_refuses_shrinking_fetch_and_preserves_file(tmp_path: Path) -> None:
    backfill_ticker(FakeProvider(n=4), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    result = backfill_ticker(
        FakeProvider(n=2), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path
    )

    assert result.guarded and not result.written
    assert len(pd.read_parquet(tmp_path / "AAPL.parquet")) == 4  # untouched


def test_force_overrides_the_guard(tmp_path: Path) -> None:
    backfill_ticker(FakeProvider(n=4), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    result = backfill_ticker(
        FakeProvider(n=2), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path, force=True
    )

    assert result.written and not result.guarded
    assert len(pd.read_parquet(tmp_path / "AAPL.parquet")) == 2  # investigated shrink applied


def test_growing_fetch_passes_the_guard(tmp_path: Path) -> None:
    backfill_ticker(FakeProvider(n=2), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    result = backfill_ticker(
        FakeProvider(n=4), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path
    )

    assert result.written and not result.guarded
    assert len(pd.read_parquet(tmp_path / "AAPL.parquet")) == 4


class CorruptProvider(OHLCVProvider):
    """Serves a frame violating the high >= low invariant."""

    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = canonical_frame(ticker)
        df.loc[0, "high"] = df.loc[0, "low"] / 2
        return df


def test_backfill_halts_on_invalid_batch_but_keeps_landing_file(tmp_path: Path) -> None:
    with pytest.raises(SchemaErrors):
        backfill_ticker(CorruptProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    # The landing zone retains the as-fetched batch for forensics; the raised
    # error is what stops anything downstream from consuming it.
    assert (tmp_path / "AAPL.parquet").exists()
