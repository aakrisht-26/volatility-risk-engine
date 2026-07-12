"""Tests for the log-variance wrapper and its retransformation.

The half-variance test is the load-bearing one: on lognormal data the wrapper
must estimate the conditional MEAN (exp(m + s2/2)), not the median (exp(m)) —
the difference is exactly the Jensen gap the v2 ruling requires us to handle
explicitly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from test_walkforward import synthetic_features
from volrisk.evaluate.walkforward import walk_forward_feature_forecasts
from volrisk.models.feature_models import make_log_har
from volrisk.models.har import HAR_FEATURES, har_coefficients
from volrisk.models.logspace import LogVarianceRegressor


def test_half_variance_correction_targets_conditional_mean() -> None:
    rng = np.random.default_rng(3)
    n = 40_000
    X = pd.DataFrame({"c": np.ones(n)})
    log_y = rng.normal(-9.0, 0.8, n)  # lognormal variance-like target
    y = pd.Series(np.exp(log_y))

    model = LogVarianceRegressor(LinearRegression()).fit(X, y)
    corrected = float(model.predict(X.iloc[[0]])[0])
    raw = float(np.exp(model.inner.predict(X.iloc[[0]])[0]))

    # exp(m + s2/2) matches the lognormal mean; exp(m) sits exp(-s2/2) ~ 27% below.
    assert corrected == pytest.approx(y.mean(), rel=0.02)
    assert raw < 0.80 * corrected


def test_predictions_are_strictly_positive_and_finite() -> None:
    df = synthetic_features()
    result = walk_forward_feature_forecasts(df, HAR_FEATURES, make_log_har, min_train=80)

    assert (result.forecasts > 0).all()
    assert np.isfinite(result.forecasts).all()


def test_floor_never_binds_with_log_space_fits() -> None:
    df = synthetic_features()
    result = walk_forward_feature_forecasts(df, HAR_FEATURES, make_log_har, min_train=80)

    assert result.floored == 0  # the canary: non-zero means investigate upstream


def test_non_positive_targets_are_dropped_not_fatal() -> None:
    rng = np.random.default_rng(5)
    n = 50
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = pd.Series(np.exp(rng.normal(-9, 0.5, n)))
    y.iloc[7] = 0.0  # degenerate flat-bar target

    model = LogVarianceRegressor(LinearRegression()).fit(X, y)

    assert (model.predict(X) > 0).all()


def test_har_coefficients_reachable_through_wrapper() -> None:
    df = synthetic_features()
    result = walk_forward_feature_forecasts(df, HAR_FEATURES, make_log_har, min_train=80)

    coefs = har_coefficients(result.final_model.inner)
    assert set(coefs) == {"const", "gk_var", "har_rv_w", "har_rv_m"}


def test_har_forecast_stays_bounded_through_a_volatility_spike() -> None:
    """Pin the v2 first-cut explosion: log-log HAR must not emit astronomical
    over-forecasts when the variance components spike by orders of magnitude."""
    df = synthetic_features()
    spike = df.index[90]
    df.loc[spike:, ["gk_var", "har_rv_w", "har_rv_m", "target_gk_var_next"]] *= 1000.0

    result = walk_forward_feature_forecasts(df, HAR_FEATURES, make_log_har, min_train=80)

    # 1.0 daily variance = ~1590% annualized vol: an absurdly generous ceiling
    # that the level-feature hybrid blew straight through.
    assert float(result.forecasts.max()) < 1.0
