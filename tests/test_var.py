"""Tests for parametric VaR, breach convention, calibration, and the backtest core.

The breach-sign test is load-bearing: VaR's classic bug is comparing the return
to +VaR instead of -VaR, which silently inverts the tail.
"""

from datetime import date

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from volrisk.risk.backtest import (
    BASE_MODEL_ORDER,
    CALIBRATION_TRIGGER_RATE,
    _per_ticker_wide,
    calibration_factors,
    evaluate_coverage,
    prediction_i_confirmed,
)
from volrisk.risk.var import TAIL_PROB, Z_SCORE, breach_mask, var_threshold


def sessions(n: int) -> list[date]:
    idx = mcal.get_calendar("XNYS").schedule("2016-01-04", "2026-12-31").index[:n]
    return [ts.date() for ts in idx]


def test_z_scores_match_standard_normal_quantiles() -> None:
    assert Z_SCORE[95] == pytest.approx(1.6448536, abs=1e-6)
    assert Z_SCORE[99] == pytest.approx(2.3263479, abs=1e-6)
    assert TAIL_PROB[95] == pytest.approx(0.05)
    assert TAIL_PROB[99] == pytest.approx(0.01)


def test_var_threshold_is_z_times_sigma() -> None:
    var = np.array([4e-4, 1e-4])  # sigma = 0.02, 0.01
    thr = var_threshold(var, 95)
    np.testing.assert_allclose(thr, Z_SCORE[95] * np.array([0.02, 0.01]))


def test_breach_is_the_lower_loss_tail_not_the_upper() -> None:
    thr = np.array([0.03, 0.03, 0.03])
    returns = np.array([-0.05, 0.05, -0.01])  # big loss, big gain, small loss
    mask = breach_mask(returns, thr)
    # Only the -0.05 loss breaches; the +0.05 gain must NOT (the off-by-sign trap).
    assert mask.tolist() == [True, False, False]


def test_coverage_recovers_nominal_rate_on_calibrated_normal_draws() -> None:
    rng = np.random.default_rng(0)
    n = 100_000
    sigma = 0.02
    idx = pd.RangeIndex(n)
    returns = pd.Series(rng.normal(0.0, sigma, n), index=idx)
    variance = pd.Series(np.full(n, sigma**2), index=idx)

    summary95, _ = evaluate_coverage(returns, variance, 95)
    summary99, _ = evaluate_coverage(returns, variance, 99)

    # With correctly specified variance the breach rate matches the tail prob.
    assert summary95["breach_rate"] == pytest.approx(0.05, abs=0.003)
    assert summary99["breach_rate"] == pytest.approx(0.01, abs=0.002)
    assert summary95["kupiec_p"] > 0.05  # correct coverage is not rejected


def test_understated_variance_produces_excess_breaches() -> None:
    rng = np.random.default_rng(1)
    n = 50_000
    true_sigma = 0.02
    returns = pd.Series(rng.normal(0.0, true_sigma, n))
    understated = pd.Series(np.full(n, (true_sigma * 0.7) ** 2))  # forecast too small

    summary, events = evaluate_coverage(returns, understated, 95)

    assert summary["breach_rate"] > 0.05  # under-coverage
    assert summary["kupiec_p"] < 0.05  # and Kupiec rejects it
    assert len(events) == summary["observed_breaches"]


def test_calibration_factors_are_monthly_steps_without_lookahead() -> None:
    dates = sessions(80)
    feat = pd.DataFrame(
        {
            "trade_date": dates,
            "r2": np.linspace(2.0, 3.0, 80),  # r2 > gk_var: overnight-gap inflation
            "gk_var": np.linspace(1.0, 1.5, 80),
        }
    )
    forecast_dates = dates[40:]  # forecast only over the back half

    c = calibration_factors(feat, forecast_dates)

    # Constant within a calendar month (a monthly step function).
    by_month = pd.Series(c.values, index=[(d.year, d.month) for d in c.index])
    assert (by_month.groupby(level=0).nunique() == 1).all()
    # No look-ahead: the factor for the earliest forecast date uses only strictly
    # earlier rows, so it cannot equal a ratio computed through that date.
    d0 = forecast_dates[0]
    train = feat[feat["trade_date"] < d0]
    assert c.iloc[0] == pytest.approx(train["r2"].mean() / train["gk_var"].mean())
    assert c.iloc[0] > 1.0  # r2 exceeds gk_var, so calibration inflates variance


def test_persisted_cal_rows_are_never_reconsumed_as_calibration_inputs() -> None:
    """The circular-calibration failure mode: _cal rows are persisted for the
    dashboard, so the backtest's base matrix must exclude them — calibrated
    variants are always derived fresh from base series + training-only factors."""
    dates = sessions(3)
    frames = [
        pd.DataFrame(
            {
                "ticker": "AAPL",
                "trade_date": dates,
                "model": model,
                "var_forecast": 2.0e-4,
                "log_return": [0.01, -0.01, 0.005],
            }
        )
        for model in [*BASE_MODEL_ORDER, "har_rv_cal", "lgbm_cal", "lgbm_vix_cal"]
    ]

    wide, _ = _per_ticker_wide(pd.concat(frames, ignore_index=True))["AAPL"]

    assert set(wide.columns) == set(BASE_MODEL_ORDER)  # no _cal column ever


def test_prediction_confirmation_criterion() -> None:
    def cov(rate: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": "AAPL", "model": m, "level": 95, "breach_rate": rate}
                for m in ("har_rv", "lgbm", "lgbm_vix")
            ]
        )

    assert prediction_i_confirmed(cov(CALIBRATION_TRIGGER_RATE + 0.005))
    assert not prediction_i_confirmed(cov(CALIBRATION_TRIGGER_RATE - 0.005))
