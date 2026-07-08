"""Tests for the backfill writer, using an in-memory fake provider. No network."""

from datetime import date
from pathlib import Path

import pandas as pd

from volrisk.ingest.backfill import backfill_ticker, parquet_path, run_backfill
from volrisk.providers.base import OHLCVProvider


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


class FakeProvider(OHLCVProvider):
    """Serves a fixed canonical frame; lets backfill be tested without network."""

    def fetch_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = canonical_frame(ticker)
        return df


def test_parquet_path_strips_index_caret(tmp_path: Path) -> None:
    assert parquet_path(tmp_path, "^GSPC") == tmp_path / "GSPC.parquet"
    assert parquet_path(tmp_path, "AAPL") == tmp_path / "AAPL.parquet"


def test_backfill_ticker_writes_parquet(tmp_path: Path) -> None:
    n = backfill_ticker(FakeProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    assert n == 2
    written = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert len(written) == 2
    assert not list(tmp_path.glob("*.tmp"))  # atomic write left no temp files


def test_backfill_rerun_is_idempotent(tmp_path: Path) -> None:
    backfill_ticker(FakeProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)
    backfill_ticker(FakeProvider(), "AAPL", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    written = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert len(written) == 2  # file replaced wholesale — zero duplicate rows
    assert written["trade_date"].is_unique


def test_run_backfill_covers_all_tickers(tmp_path: Path) -> None:
    counts = run_backfill(FakeProvider(), ("^GSPC", "AAPL"), years=1, out_dir=tmp_path)

    assert counts == {"^GSPC": 2, "AAPL": 2}
    assert (tmp_path / "GSPC.parquet").exists()
    assert (tmp_path / "AAPL.parquet").exists()
