"""Run the feature-based ladder walk-forward: HAR-RV, LightGBM, LightGBM+VIX.

Usage::

    uv run python -m volrisk.models.feature_models [--tickers ...]

Model tags: ``har_rv``, ``lgbm``, ``lgbm_vix``. The lgbm_vix variant appends
two exogenous regressors derived from ^VIX reference data (per the CLAUDE.md
^VIX ruling: candidate exogenous feature, never a modeled asset): the VIX
close and its one-day log change, both taken from the feature row's own
session — which is d-1 relative to the session being forecast, so the
information lag is >= 1 by construction. They are computed at run time from
clean bars and are deliberately NOT part of the features schema.

All three models fit LOG variance and retransform with the lognormal
half-variance correction (ablation v2 ruling; see models/logspace.py for the
choice and its rationale). The floored-predictions count is therefore a pure
canary: log-space predictions are positive by construction, so the expected
count is 0 — a non-zero value means something upstream is wrong and must be
investigated, not shrugged at.

Prints HAR-RV's fitted coefficients (log-variance space) at the final refit
per ticker.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine
from volrisk.db.loaders import upsert_variance_forecasts
from volrisk.evaluate.walkforward import walk_forward_feature_forecasts
from volrisk.models.baselines import MIN_TRAIN_SESSIONS, NON_MODELED_TICKERS, forecasts_frame
from volrisk.models.har import HAR_FEATURES, har_coefficients, make_har_model
from volrisk.models.lgbm import LGBM_FEATURES, make_lgbm_model
from volrisk.models.logspace import LogVarianceRegressor
from volrisk.transform.cleaning import next_session

logger = logging.getLogger(__name__)

HAR_TAG = "har_rv"
LGBM_TAG = "lgbm"
LGBM_VIX_TAG = "lgbm_vix"
VIX_COLS: tuple[str, ...] = ("vix_close", "vix_log_change")


def make_log_har() -> LogVarianceRegressor:
    return LogVarianceRegressor(make_har_model())


def make_log_lgbm() -> LogVarianceRegressor:
    return LogVarianceRegressor(make_lgbm_model())


def load_features(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT * FROM features.daily_features ORDER BY ticker, trade_date"), engine
    )


def load_vix_reference(engine: Engine) -> pd.DataFrame:
    vix = pd.read_sql_query(
        text(
            "SELECT trade_date, close AS vix_close FROM clean.daily_bars"
            " WHERE ticker = '^VIX' ORDER BY trade_date"
        ),
        engine,
    )
    vix["vix_log_change"] = np.log(vix["vix_close"]).diff()
    return vix


def run_feature_models(
    engine: Engine,
    tickers: list[str] | None = None,
    min_train: int = MIN_TRAIN_SESSIONS,
) -> pd.DataFrame:
    feats = load_features(engine)
    vix = load_vix_reference(engine)
    if tickers is None:
        tickers = [t for t in feats["ticker"].unique() if t not in NON_MODELED_TICKERS]

    rows = []
    for ticker in tickers:
        frame = feats[feats["ticker"] == ticker]
        frame_vix = frame.merge(vix, on="trade_date", how="left")
        for tag, cols, data, factory in (
            (HAR_TAG, HAR_FEATURES, frame, make_log_har),
            (LGBM_TAG, LGBM_FEATURES, frame, make_log_lgbm),
            (LGBM_VIX_TAG, (*LGBM_FEATURES, *VIX_COLS), frame_vix, make_log_lgbm),
        ):
            result = walk_forward_feature_forecasts(data, cols, factory, min_train=min_train)
            n = upsert_variance_forecasts(
                engine, forecasts_frame(ticker, result.forecasts, tag), context=f"{ticker}:{tag}"
            )
            if result.live_forecast is not None:
                live_date = next_session(data["trade_date"].max())
                upsert_variance_forecasts(
                    engine,
                    forecasts_frame(
                        ticker, pd.Series([result.live_forecast], index=[live_date]), tag
                    ),
                    context=f"{ticker}:{tag}:live",
                    is_live=True,
                )
            row = {
                "ticker": ticker,
                "model": tag,
                "rows": n,
                "first": result.forecasts.index[0],
                "last": result.forecasts.index[-1],
                "refits": result.refits,
                "floored": result.floored,
            }
            if tag == HAR_TAG and result.final_model is not None:
                coefs = har_coefficients(result.final_model.inner)
                logger.info("%s HAR-RV final-refit coefficients (log space): %s", ticker, coefs)
                row["har_coefs"] = coefs
            rows.append(row)
            logger.info("%s %s: %d forecasts", ticker, tag, n)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward HAR-RV and LightGBM models.")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    summary = run_feature_models(get_engine(), args.tickers)

    print("\n=== Feature-model forecasts written to forecasts.daily_variance ===")
    print(
        summary[["ticker", "model", "rows", "first", "last", "refits", "floored"]].to_string(
            index=False
        )
    )
    print(f"\ntotal rows upserted: {summary['rows'].sum()}")
    floored = int(summary["floored"].sum())
    print(
        f"floored predictions (canary, expected 0 with log-space fits): {floored}"
        + ("  << INVESTIGATE" if floored else "")
    )

    har_rows = summary[summary["model"] == HAR_TAG]
    print("\nHAR-RV coefficients at final refit (log-variance space):")
    for _, row in har_rows.iterrows():
        coefs = ", ".join(f"{k}={v:.3e}" for k, v in row["har_coefs"].items())
        print(f"  {row['ticker']:>6}: {coefs}")


if __name__ == "__main__":
    main()
