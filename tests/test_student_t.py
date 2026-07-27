"""Tests for the Student-t VaR layer.

The load-bearing property is the variance match: a t scaled so its variance
equals the model's forecast must change the tail SHAPE only. That has a
counter-intuitive consequence worth pinning — at 95% the variance-matched t
quantile is SMALLER than the normal z, because mass moved from the shoulders
to the tails.
"""

from datetime import date

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from scipy import stats

from volrisk.risk.student_t import (
    MAX_DF,
    MIN_DF,
    fit_t_df,
    t_quantile_scaled,
    t_var_thresholds,
    walk_forward_t_df,
)
from volrisk.risk.var import Z_SCORE


def sessions(n: int) -> list[date]:
    idx = mcal.get_calendar("XNYS").schedule("2016-01-04", "2026-12-31").index[:n]
    return [ts.date() for ts in idx]


# --- the scaled quantile ---


def test_scaled_quantile_matches_the_variance_definition() -> None:
    df = 6.0
    expected = stats.t.ppf(0.99, df) / np.sqrt(df / (df - 2))

    assert t_quantile_scaled(df, 0.99) == pytest.approx(expected, rel=1e-12)


def test_high_df_converges_to_the_normal_z() -> None:
    assert t_quantile_scaled(1e6, 0.95) == pytest.approx(Z_SCORE[95], abs=1e-4)
    assert t_quantile_scaled(1e6, 0.99) == pytest.approx(Z_SCORE[99], abs=1e-4)


def test_variance_matched_t_is_fatter_at_99_and_thinner_at_95() -> None:
    """Mass moves from shoulders to tails: the 95% threshold SHRINKS.

    This is why a t variant can improve 99% coverage while making 95% coverage
    worse — the two levels move in opposite directions by construction.
    """
    for df in (3.0, 5.0, 8.0):
        assert t_quantile_scaled(df, 0.99) > Z_SCORE[99]
        assert t_quantile_scaled(df, 0.95) < Z_SCORE[95]


def test_df_at_or_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite variance"):
        t_quantile_scaled(2.0, 0.99)


# --- df estimation ---


def test_mle_recovers_known_degrees_of_freedom() -> None:
    rng = np.random.default_rng(11)
    z = rng.standard_t(5, 20_000)

    assert fit_t_df(z) == pytest.approx(5.0, rel=0.15)


def test_gaussian_data_pushes_df_to_the_upper_bound() -> None:
    rng = np.random.default_rng(3)

    assert fit_t_df(rng.normal(size=20_000)) > 20  # no fat tails to find


def test_too_few_observations_falls_back_to_normal_equivalent() -> None:
    assert fit_t_df(np.array([0.1, -0.2, 0.3])) == MAX_DF


def test_extremely_fat_data_is_clamped_above_two() -> None:
    rng = np.random.default_rng(5)
    z = rng.standard_t(1.2, 5_000)  # df < 2: no finite variance

    assert fit_t_df(z) >= MIN_DF


# --- walk-forward estimation ---


def test_walk_forward_df_is_a_monthly_step_function_without_lookahead() -> None:
    n = 400
    idx = sessions(n)
    rng = np.random.default_rng(7)
    variance = pd.Series(np.full(n, 4e-4), index=idx)
    returns = pd.Series(rng.standard_t(4, n) * 0.02 / np.sqrt(2.0), index=idx)

    df_series = walk_forward_t_df(returns, variance, min_train=100)

    by_month = pd.Series(df_series.values, index=[(d.year, d.month) for d in df_series.index])
    assert (by_month.groupby(level=0).nunique() == 1).all()  # constant within a month
    assert (df_series.iloc[:100] == MAX_DF).all()  # no estimate before min_train
    assert df_series.iloc[-1] < MAX_DF  # a real fit happened later


def test_thresholds_scale_with_sigma_and_use_per_date_df() -> None:
    idx = sessions(2)
    variance = pd.Series([4e-4, 1e-4], index=idx)  # sigma = 0.02, 0.01
    df_series = pd.Series([5.0, 8.0], index=idx)

    thr = t_var_thresholds(variance, df_series, 0.99)

    assert thr[0] == pytest.approx(0.02 * t_quantile_scaled(5.0, 0.99), rel=1e-12)
    assert thr[1] == pytest.approx(0.01 * t_quantile_scaled(8.0, 0.99), rel=1e-12)


def test_t_thresholds_beat_normal_at_99_on_the_same_variance() -> None:
    from volrisk.risk.var import var_threshold

    idx = sessions(3)
    variance = pd.Series(np.full(3, 4e-4), index=idx)
    df_series = pd.Series(np.full(3, 5.0), index=idx)

    assert (t_var_thresholds(variance, df_series, 0.99) > var_threshold(variance, 99)).all()
