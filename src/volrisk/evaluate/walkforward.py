"""Shared walk-forward harness for feature-based variance models.

Semantics (identical to the Step-7 baselines): a forecast row is keyed by the
session being forecast. The prediction for session d uses the feature row of
session d-1 (available after d-1's close); the model was last refit at a
calendar-month boundary using only training pairs whose TARGET session
precedes that month's first forecast. Expanding window, minimum ``min_train``
feature rows before the first forecast, refit monthly (CLAUDE.md).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Predictions are floored here: QLIKE and the DB CHECK require positive
#: variance, and unconstrained regressors can emit values <= 0. The floor is a
#: daily variance of 1e-8 (~0.16% annualized vol); floored counts are reported
#: canary-style by the runner.
VARIANCE_FLOOR = 1e-8


class VarianceRegressor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class WalkForwardResult:
    forecasts: pd.Series
    refits: int = 0
    floored: int = 0
    skipped_incomplete: int = 0
    final_model: VarianceRegressor | None = None
    #: Prediction from the LAST feature row — the forecast for the next session
    #: after the data ends (the "live" row; no realized outcome exists yet).
    live_forecast: float | None = None


def walk_forward_feature_forecasts(
    features: pd.DataFrame,
    feature_cols: tuple[str, ...],
    model_factory: Callable[[], VarianceRegressor],
    target_col: str = "target_gk_var_next",
    min_train: int = 756,
) -> WalkForwardResult:
    """Walk-forward forecasts for one ticker's feature frame.

    No-leakage arithmetic: at a refit before forecasting session ``d = D[i]``,
    training rows are ``j <= i-2`` — row j's target references session j+1
    <= i-1, so no training pair touches session d or later. The prediction for
    session d uses row i-1's features, known at d-1's close. Training rows
    with any missing feature or target (warm-up NULLs) are dropped.
    """
    df = features.sort_values("trade_date").reset_index(drop=True)
    dates = list(df["trade_date"])
    n = len(df)
    result = WalkForwardResult(forecasts=pd.Series(dtype=float))
    if n <= min_train:
        return result

    cols = list(feature_cols)
    X_all = df[cols]
    model: VarianceRegressor | None = None
    refit_month: tuple[int, int] | None = None
    out_dates: list = []
    out_vals: list[float] = []

    for i in range(min_train, n):
        month = (dates[i].year, dates[i].month)
        if model is None or month != refit_month:
            train = df.iloc[: i - 1]
            usable = train[cols].notna().all(axis=1) & train[target_col].notna()
            model = model_factory()
            model.fit(train.loc[usable, cols], train.loc[usable, target_col])
            result.refits += 1
            refit_month = month
        x = X_all.iloc[[i - 1]]
        if bool(x.isna().any(axis=1).iloc[0]):
            result.skipped_incomplete += 1
            continue
        pred = float(model.predict(x)[0])
        if pred < VARIANCE_FLOOR:
            pred = VARIANCE_FLOOR
            result.floored += 1
        out_dates.append(dates[i])
        out_vals.append(pred)

    result.forecasts = pd.Series(out_vals, index=out_dates, dtype=float)
    result.final_model = model

    # Live next-session forecast: predict from the LAST feature row (session
    # n-1's features, known at its close) — the forecast for the session that
    # follows the data. Floored like every prediction; the floor canary counts it.
    x_last = X_all.iloc[[n - 1]]
    if model is not None and not bool(x_last.isna().any(axis=1).iloc[0]):
        live = float(model.predict(x_last)[0])
        if live < VARIANCE_FLOOR:
            live = VARIANCE_FLOOR
            result.floored += 1
        result.live_forecast = live
    return result
