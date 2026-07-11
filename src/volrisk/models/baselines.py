"""Run the baseline ladder — EWMA(0.94) and GARCH(1,1) — walk-forward, store forecasts.

Usage::

    uv run python -m volrisk.models.baselines [--tickers ...]

Walk-forward configuration per CLAUDE.md: expanding window, minimum ~3 years
of training sessions (756), GARCH refit monthly. Forecast rows are keyed by
the session being forecast; ``var_forecast`` is daily variance in RETURN
units (see db/migrations/004_forecasts.sql).

^VIX is excluded by default: it is reference data and is never modeled as a
tradable asset (CLAUDE.md data decisions). Pass --tickers to override.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine
from volrisk.db.loaders import upsert_variance_forecasts
from volrisk.models.ewma import ewma_variance_forecasts
from volrisk.models.garch import garch_variance_forecasts

logger = logging.getLogger(__name__)

EWMA_TAG = "ewma_094"
GARCH_TAG = "garch_11"

#: ~3 years of NYSE sessions (CLAUDE.md: minimum ~3y train).
MIN_TRAIN_SESSIONS = 756

NON_MODELED_TICKERS = ("^VIX",)


def load_clean_returns(engine: Engine) -> pd.DataFrame:
    """All clean per-ticker log returns (first row per ticker is NULL and dropped)."""
    return pd.read_sql_query(
        text(
            "SELECT ticker, trade_date, log_return FROM clean.daily_bars"
            " WHERE log_return IS NOT NULL ORDER BY ticker, trade_date"
        ),
        engine,
    )


def forecasts_frame(ticker: str, series: pd.Series, model: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": list(series.index),
            "model": model,
            "var_forecast": series.to_numpy(dtype=float),
        }
    )


def run_baselines(
    engine: Engine,
    tickers: list[str] | None = None,
    min_train: int = MIN_TRAIN_SESSIONS,
) -> pd.DataFrame:
    """Produce and upsert both baseline forecast series; returns a summary frame."""
    returns = load_clean_returns(engine)
    if tickers is None:
        tickers = [t for t in returns["ticker"].unique() if t not in NON_MODELED_TICKERS]

    rows = []
    for ticker in tickers:
        r = returns[returns["ticker"] == ticker].set_index("trade_date")["log_return"]
        garch_result = garch_variance_forecasts(r, min_train=min_train)
        for model, series, convergence in (
            (EWMA_TAG, ewma_variance_forecasts(r, min_train=min_train), None),
            (GARCH_TAG, garch_result.forecasts, garch_result),
        ):
            n = upsert_variance_forecasts(
                engine, forecasts_frame(ticker, series, model), context=f"{ticker}:{model}"
            )
            row = {
                "ticker": ticker,
                "model": model,
                "rows": n,
                "first": series.index[0],
                "last": series.index[-1],
                "refits": convergence.refits if convergence else 0,
                "fallbacks": convergence.fallback_refits if convergence else 0,
                "unconverged": convergence.unconverged_consumed if convergence else 0,
            }
            rows.append(row)
            logger.info(
                "%s %s: %d forecasts (%s..%s)", ticker, model, n, series.index[0], series.index[-1]
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward EWMA and GARCH baselines.")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    summary = run_baselines(get_engine(), args.tickers)

    print("\n=== Baseline forecasts written to forecasts.daily_variance ===")
    print(summary.to_string(index=False))
    print(f"\ntotal rows upserted: {summary['rows'].sum()}")

    # Convergence canary (GARCH rows only): any non-zero fallback/unconverged
    # count means the policy in volrisk.models.garch was exercised — see logs.
    garch = summary[summary["model"] == GARCH_TAG]
    print(
        f"GARCH convergence: {garch['refits'].sum()} refits, "
        f"{garch['fallbacks'].sum()} fallback(s) to previous parameters, "
        f"{garch['unconverged'].sum()} unconverged first fit(s) consumed"
    )


if __name__ == "__main__":
    main()
