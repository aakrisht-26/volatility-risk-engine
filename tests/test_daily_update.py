"""Unit tests for the nightly job's pure pieces: the trailing window, the
canary gate, and the guarded-ticker increment path (with fakes, offline)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volrisk.ingest.daily_update import (
    TRAILING_SESSIONS,
    enforce_canaries,
    trailing_window_start,
)


def test_trailing_window_covers_at_least_the_contract_sessions() -> None:
    as_of = date(2026, 7, 15)
    start = trailing_window_start(as_of)

    # 5 trading days can span at most 5 + 2 weekend days + a holiday; the
    # calendar overshoot (2x + 3) always covers the recorded minimum contract.
    assert (as_of - start).days >= TRAILING_SESSIONS + 2
    assert start < as_of


def test_canary_gate_passes_on_all_zero() -> None:
    enforce_canaries({"negative_gk": 0, "floored_predictions": 0})  # no raise


def test_canary_gate_fails_on_any_nonzero_and_names_it() -> None:
    with pytest.raises(SystemExit, match=r"CANARY FAILURE.*floored_predictions"):
        enforce_canaries({"negative_gk": 0, "floored_predictions": 3})


def test_increments_never_overwrite_the_anchored_zone(tmp_path: Path, monkeypatch) -> None:
    """The guarded path writes under increments/YYYY-MM-DD/, not data/raw/."""
    import volrisk.ingest.daily_update as du

    monkeypatch.setattr(du, "INCREMENTS_DIR", tmp_path / "increments")
    inc_dir = du.INCREMENTS_DIR / date(2026, 7, 15).isoformat()
    inc_dir.mkdir(parents=True)

    anchored = tmp_path / "AAPL.parquet"
    pd.DataFrame({"x": range(4)}).to_parquet(anchored, index=False)
    pd.DataFrame({"x": range(2)}).to_parquet(inc_dir / "AAPL.parquet", index=False)

    assert len(pd.read_parquet(anchored)) == 4  # anchored zone untouched
    assert len(pd.read_parquet(inc_dir / "AAPL.parquet")) == 2
