"""Nightly job: fetch -> validate -> load -> clean -> features -> forecasts -> VaR.

Usage::

    uv run python -m volrisk.ingest.daily_update [--force]

Design (Step-11 ruling):

- Fetch is the FULL anchored backfill (BACKFILL_START..today) through a
  yfinance -> Stooq fallback chain. This supersets the recorded minimum
  contract (re-fetch a ~5-trading-day trailing window, never only
  "yesterday") and keeps the landing-zone == DB invariant true every night.
- The monotonic guard refuses shrinking parquet overwrites. Guarded tickers
  fall back to a trailing-window fetch landed as dated increment files under
  ``data/raw/increments/YYYY-MM-DD/`` — never overwriting the anchored zone —
  and upserted directly. Self-heal after missed nights is this same path's
  big brother: the full anchored fetch always covers any gap.
- Cleaning, features, and ALL forecasts re-run in full: idempotent upserts,
  minutes of compute, and every canary executes nightly. Revisit incremental
  only if total job time exceeds ~20 minutes (recorded threshold).
- CANARIES ARE EXIT CODES (required): telescoping failures, negative gk_var,
  floored predictions, GARCH fallback refits, unconverged first fits — any
  non-zero value fails the job after the summary prints. Guarded tickers are
  surfaced in the summary as a warning (their fallback path already ran).
- The cloud DB is the system of record; a runner's parquet is deterministic
  staging reconstructable from the anchor. The durable replay copy lives on
  the dev machine (see CLAUDE.md landing-zone semantics).
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine
from volrisk.db.loaders import (
    read_landing_parquet,
    store_var_results,
    upsert_ablation_metrics,
    upsert_daily_bars,
)
from volrisk.db.migrate import apply_migrations
from volrisk.evaluate.ablation import compute_ablation
from volrisk.features.build import build_features
from volrisk.ingest.backfill import (
    BACKFILL_START,
    DEFAULT_OUT_DIR,
    PHASE_1_TICKERS,
    backfill_ticker,
    parquet_path,
)
from volrisk.models.baselines import run_baselines
from volrisk.models.feature_models import run_feature_models
from volrisk.providers.base import FallbackProvider
from volrisk.providers.stooq_provider import StooqProvider
from volrisk.providers.yfinance_provider import YFinanceProvider
from volrisk.risk.backtest import compute_backtest
from volrisk.transform.cleaning import TELESCOPE_TOLERANCE, run_cleaning
from volrisk.transform.returns import telescoping_check

logger = logging.getLogger(__name__)

#: Recorded minimum contract: ~5 trading days re-fetched on the fallback path.
TRAILING_SESSIONS = 5
INCREMENTS_DIR = DEFAULT_OUT_DIR / "increments"


def trailing_window_start(as_of: date, sessions: int = TRAILING_SESSIONS) -> date:
    """Calendar start covering at least ``sessions`` trading days (overshoot is
    harmless: everything downstream is an upsert on the natural key)."""
    return as_of - timedelta(days=sessions * 2 + 3)


def make_provider() -> FallbackProvider:
    return FallbackProvider([("yfinance", YFinanceProvider()), ("stooq", StooqProvider())])


def fetch_and_load(engine: Engine, provider: FallbackProvider, force: bool) -> pd.DataFrame:
    """Full anchored fetch per ticker; guarded tickers take the trailing-window
    increment path. Returns a per-ticker summary frame."""
    today = date.today()
    rows = []
    for ticker in PHASE_1_TICKERS:
        result = backfill_ticker(
            provider, ticker, BACKFILL_START, today, DEFAULT_OUT_DIR, force=force
        )
        if result.guarded:
            inc_dir = INCREMENTS_DIR / today.isoformat()
            inc_dir.mkdir(parents=True, exist_ok=True)
            df = provider.fetch_daily_ohlcv(ticker, trailing_window_start(today), today)
            df.to_parquet(parquet_path(inc_dir, ticker), index=False)
            loaded = upsert_daily_bars(
                engine,
                df,
                context=f"increment:{ticker}",
                source=provider.sources.get(ticker, "yfinance"),
            )
            logger.warning(
                "%s: guard fired; loaded %d trailing-window rows as increment", ticker, loaded
            )
        else:
            df = read_landing_parquet(parquet_path(DEFAULT_OUT_DIR, ticker))
            loaded = upsert_daily_bars(
                engine, df, context=ticker, source=provider.sources.get(ticker, "yfinance")
            )
        rows.append(
            {
                "ticker": ticker,
                "rows_loaded": loaded,
                "source": provider.sources.get(ticker, "?"),
                "guarded": result.guarded,
            }
        )
    return pd.DataFrame(rows)


def telescoping_failures(engine: Engine) -> list[str]:
    """Per-ticker telescoping identity over clean.daily_bars; returns failures."""
    clean = pd.read_sql_query(
        text(
            "SELECT ticker, trade_date, adj_close, log_return"
            " FROM clean.daily_bars ORDER BY ticker, trade_date"
        ),
        engine,
    )
    failed = []
    for ticker, group in clean.groupby("ticker", sort=True):
        total, endpoints = telescoping_check(group)
        if abs(total - endpoints) >= TELESCOPE_TOLERANCE:
            failed.append(str(ticker))
    return failed


def negative_gk_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM features.daily_features WHERE gk_var < 0")
        ).scalar_one()


def enforce_canaries(canaries: dict[str, int]) -> None:
    """Any non-zero canary fails the job (Step-11 ruling: canaries = exit codes)."""
    tripped = {name: value for name, value in canaries.items() if value}
    if tripped:
        raise SystemExit(f"CANARY FAILURE: {tripped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly incremental pipeline run.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="override the monotonic landing-zone guard (only after investigating)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    engine = get_engine()
    timings: dict[str, float] = {}

    def stage(name: str):
        timings[name] = time.perf_counter()
        logger.info("=== stage: %s ===", name)

    stage("migrate")
    applied = apply_migrations(engine)
    timings["migrate"] = time.perf_counter() - timings["migrate"]

    stage("fetch+load")
    provider = make_provider()
    fetch_summary = fetch_and_load(engine, provider, force=args.force)
    timings["fetch+load"] = time.perf_counter() - timings["fetch+load"]

    stage("clean")
    reports = run_cleaning(engine)
    telescope_failed = telescoping_failures(engine)
    timings["clean"] = time.perf_counter() - timings["clean"]

    stage("features")
    feature_rows = build_features(engine)
    negative_gk = negative_gk_count(engine)
    timings["features"] = time.perf_counter() - timings["features"]

    stage("baselines")
    base = run_baselines(engine)
    timings["baselines"] = time.perf_counter() - timings["baselines"]

    stage("feature-models")
    feat_models = run_feature_models(engine)
    timings["feature-models"] = time.perf_counter() - timings["feature-models"]

    stage("evaluate")
    metrics = compute_ablation(engine)
    upsert_ablation_metrics(engine, metrics)
    coverage, breaches, _ = compute_backtest(engine)
    store_var_results(engine, coverage, breaches)
    timings["evaluate"] = time.perf_counter() - timings["evaluate"]

    canaries = {
        "telescoping_failures": len(telescope_failed),
        "negative_gk": int(negative_gk),
        "garch_fallback_refits": int(base["fallbacks"].sum()),
        "garch_unconverged_consumed": int(base["unconverged"].sum()),
        "floored_predictions": int(feat_models["floored"].sum()),
    }

    print("\n=== Nightly job summary ===")
    print(f"migrations applied: {applied if applied else 'none (up to date)'}")
    print(fetch_summary.to_string(index=False))
    partials = sum(len(r.partial_rows) for r in reports)
    print(f"clean: {sum(r.clean_rows for r in reports)} rows ({partials} partial excluded)")
    print(
        f"features: {feature_rows} rows | forecasts: {base['rows'].sum()} baseline"
        f" + {feat_models['rows'].sum()} feature-model"
    )
    print(f"ablation rows: {len(metrics)} | coverage rows: {len(coverage)}")
    print("timings (s): " + ", ".join(f"{k}={v:.1f}" for k, v in timings.items()))
    if fetch_summary["guarded"].any():
        print(
            f"WARNING guarded tickers (increment path used): "
            f"{fetch_summary.loc[fetch_summary.guarded, 'ticker'].tolist()}"
        )
    print(f"canaries: {canaries}")

    enforce_canaries(canaries)
    print("nightly job OK")


if __name__ == "__main__":
    main()
