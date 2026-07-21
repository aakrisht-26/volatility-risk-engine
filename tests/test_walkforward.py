"""Tests for the shared walk-forward harness and the feature models.

The alignment test is the load-bearing one: an identity model that predicts
its input's gk_var proves that the forecast for session d is built from the
feature row of session d-1 — the exact off-by-one that walk-forward code gets
wrong. Leakage is tested with a real regression whose fit would change if the
future leaked into training.
"""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from volrisk.evaluate.walkforward import VARIANCE_FLOOR, walk_forward_feature_forecasts
from volrisk.features.crosscheck import recompute_features_pandas
from volrisk.models.har import HAR_FEATURES, har_coefficients, make_har_model
from volrisk.models.lgbm import LGBM_FEATURES, make_lgbm_model
from volrisk.transform.returns import add_log_returns


def synthetic_features(n: int = 120, seed: int = 7, ticker: str = "SYN") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = mcal.get_calendar("XNYS").schedule("2024-01-02", "2024-12-31").index[:n]
    rets = rng.normal(0.0, 0.02, n)
    rets[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0.0, 0.003, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.004, n)))
    clean = pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": [ts.date() for ts in sessions],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": pd.array([1_000_000] * n, dtype="Int64"),
        }
    )
    return recompute_features_pandas(add_log_returns(clean))


class IdentityGK:
    """Predicts the input row's gk_var — pins prediction-row alignment."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["gk_var"].to_numpy()


class AlwaysNegative:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), -1.0)


def test_prediction_for_session_d_uses_features_of_d_minus_1() -> None:
    df = synthetic_features()
    result = walk_forward_feature_forecasts(df, ("gk_var",), IdentityGK, min_train=80)

    by_date = df.set_index("trade_date")["gk_var"]
    dates = list(df["trade_date"])
    for forecast_date, value in result.forecasts.items():
        prev_date = dates[dates.index(forecast_date) - 1]
        assert value == by_date.loc[prev_date]


def test_walkforward_training_never_sees_the_future() -> None:
    df = synthetic_features()
    out = walk_forward_feature_forecasts(df, HAR_FEATURES, make_har_model, min_train=80).forecasts

    k = 100
    mutated = df.copy()
    cols = [*HAR_FEATURES, "target_gk_var_next"]
    mutated.loc[k:, cols] = mutated.loc[k:, cols] * 7.5  # corrupt the future
    out_mutated = walk_forward_feature_forecasts(
        mutated, HAR_FEATURES, make_har_model, min_train=80
    ).forecasts

    # Forecast for session d uses features at d-1 and training targets through
    # d-1: every forecast strictly before session k is untouched.
    cutoff = df["trade_date"].iloc[k]
    pd.testing.assert_series_equal(out[out.index < cutoff], out_mutated[out_mutated.index < cutoff])


def test_walkforward_refits_monthly_and_floors_negative_predictions() -> None:
    df = synthetic_features()  # 120 sessions: Jan..Jun 2024
    result = walk_forward_feature_forecasts(df, ("gk_var",), AlwaysNegative, min_train=80)

    assert result.refits == 3  # Apr (first forecast), May, Jun boundaries
    # Every historical prediction floored, plus the live next-session one.
    assert result.floored == len(result.forecasts) + 1
    assert (result.forecasts == VARIANCE_FLOOR).all()
    assert result.live_forecast == VARIANCE_FLOOR


def test_live_forecast_predicts_from_the_last_feature_row() -> None:
    df = synthetic_features()
    result = walk_forward_feature_forecasts(df, ("gk_var",), IdentityGK, min_train=80)

    # Identity model: the live next-session forecast IS the final row's gk_var.
    assert result.live_forecast == df["gk_var"].iloc[-1]


def test_har_model_exposes_named_coefficients() -> None:
    df = synthetic_features()
    result = walk_forward_feature_forecasts(df, HAR_FEATURES, make_har_model, min_train=80)

    coefs = har_coefficients(result.final_model)
    assert set(coefs) == {"const", "gk_var", "har_rv_w", "har_rv_m"}


def test_lgbm_factory_is_deterministic() -> None:
    df = synthetic_features()

    def run() -> pd.Series:
        return walk_forward_feature_forecasts(
            df, LGBM_FEATURES, make_lgbm_model, min_train=100
        ).forecasts

    first, second = run(), run()

    pd.testing.assert_series_equal(first, second)
    assert (first > 0).all() and np.isfinite(first).all()


def test_walkforward_too_short_frame_yields_empty() -> None:
    df = synthetic_features(n=50)
    result = walk_forward_feature_forecasts(df, HAR_FEATURES, make_har_model, min_train=50)
    assert result.forecasts.empty


def test_qlike_and_rmse_hand_values() -> None:
    from volrisk.evaluate.metrics import qlike, rmse_ann_vol_pct

    h = np.array([2e-4, 2e-4])
    assert qlike(h, h) == pytest.approx(0.0, abs=1e-15)
    # h/f = 2 everywhere: qlike = 2 - ln 2 - 1
    assert qlike(h, h / 2) == pytest.approx(1.0 - np.log(2.0), rel=1e-12)
    assert rmse_ann_vol_pct(h, h) == pytest.approx(0.0, abs=1e-12)
    # constant 20% realized vs 22% forecast annualized vol -> RMSE exactly 2.0 points
    realized = np.full(3, 0.20**2 / 252)
    forecast = np.full(3, 0.22**2 / 252)
    assert rmse_ann_vol_pct(realized, forecast) == pytest.approx(2.0, rel=1e-12)
