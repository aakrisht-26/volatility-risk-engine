"""HAR-RV in Corsi's log-log form: OLS of ln(RV_next) on ln(RV_d/w/m).

The components are the features layer's Garman-Klass daily variance and its
5- and 22-day means. BOTH sides are in logs (the classic log-HAR): with level
features and a log target, spike-day component values multiplied by large
level-space coefficients would land in the exponent and produce astronomical
over-forecasts — observed in ablation v2's first cut (RMSE in the millions of
vol points while QLIKE, with its logarithmic over-forecast penalty, barely
moved). In log-log space the coefficients are ~0.2-0.5 elasticities and
extrapolation stays linear. The log-variance TARGET handling (half-variance
retransformation) lives in models/logspace.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

HAR_FEATURES: tuple[str, ...] = ("gk_var", "har_rv_w", "har_rv_m")


def _safe_log(x):
    # gk_var is provably non-negative but can be exactly zero on a degenerate
    # flat bar; clip so a lone zero cannot produce -inf in the design matrix.
    return np.log(np.maximum(x, 1e-12))


def make_har_model() -> Pipeline:
    return make_pipeline(
        FunctionTransformer(_safe_log, feature_names_out="one-to-one"), LinearRegression()
    )


def har_coefficients(model: Pipeline | LinearRegression) -> dict[str, float]:
    """Named coefficients of a fitted HAR model (log-log space: elasticities)."""
    ols = model[-1] if isinstance(model, Pipeline) else model
    out = {"const": float(ols.intercept_)}
    out.update({name: float(c) for name, c in zip(HAR_FEATURES, ols.coef_, strict=True)})
    return out
