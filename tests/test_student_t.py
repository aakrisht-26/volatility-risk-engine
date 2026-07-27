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

    fit = fit_t_df(z)

    assert fit.df == pytest.approx(5.0, rel=0.15)
    assert fit.status == "interior"


def test_gaussian_data_clamps_high_and_says_so() -> None:
    """The failure mode the addendum exposed: Gaussian residuals send the MLE
    to the boundary, where the t variant is just the normal one again."""
    rng = np.random.default_rng(3)

    fit = fit_t_df(rng.normal(size=20_000))

    assert fit.status == "clamped_high"
    assert fit.df == MAX_DF
    assert fit.raw_df >= MAX_DF  # the raw estimate is retained for the report


def test_too_few_observations_falls_back_to_normal_equivalent() -> None:
    fit = fit_t_df(np.array([0.1, -0.2, 0.3]))

    assert fit.df == MAX_DF
    assert fit.status == "insufficient_data"


def test_extremely_fat_data_clamps_low_and_says_so() -> None:
    rng = np.random.default_rng(5)
    z = rng.standard_t(1.2, 5_000)  # df < 2: no finite variance

    fit = fit_t_df(z)

    assert fit.status == "clamped_low"
    assert fit.df == MIN_DF
    assert fit.raw_df < MIN_DF


def test_a_low_clamp_is_rejected_not_consumed() -> None:
    """The Step-7 remedy applied here: a df <= 2 fit is a FAILED fit, so the
    walk-forward must keep the previous month's df rather than emit a VaR
    smaller than the Gaussian one."""
    n = 300
    idx = sessions(n)
    rng = np.random.default_rng(2)
    variance = pd.Series(np.full(n, 4e-4), index=idx)
    z = rng.standard_t(6, n)
    z[200:] = rng.standard_t(1.1, n - 200) * 8  # force an infinite-variance window
    returns = pd.Series(z * 0.02, index=idx)

    est = walk_forward_t_df(returns, variance, min_train=100)

    assert est.n_clamped_low > 0  # the failure happened and was counted
    assert (est.df_series > MIN_DF).all()  # but no month ever used the collapsed df


def test_low_clamp_collapses_the_quantile_below_the_normal() -> None:
    """Why a low clamp is dangerous rather than merely inelegant: holding the
    variance fixed as df approaches 2 drives the multiplier toward zero, so the
    'fat-tailed' VaR ends up far SMALLER than the Gaussian one."""
    assert t_quantile_scaled(MIN_DF, 0.99) < Z_SCORE[99] / 2
    assert t_quantile_scaled(MIN_DF, 0.95) < Z_SCORE[95] / 3


# --- walk-forward estimation ---


def test_walk_forward_df_is_a_monthly_step_function_without_lookahead() -> None:
    n = 400
    idx = sessions(n)
    rng = np.random.default_rng(7)
    variance = pd.Series(np.full(n, 4e-4), index=idx)
    returns = pd.Series(rng.standard_t(4, n) * 0.02 / np.sqrt(2.0), index=idx)

    est = walk_forward_t_df(returns, variance, min_train=100)
    df_series = est.df_series

    by_month = pd.Series(df_series.values, index=[(d.year, d.month) for d in df_series.index])
    assert (by_month.groupby(level=0).nunique() == 1).all()  # constant within a month
    assert (df_series.iloc[:100] == MAX_DF).all()  # no estimate before min_train
    assert df_series.iloc[-1] < MAX_DF  # a real fit happened later
    # Diagnostics accompany the series so clamping can never pass unnoticed.
    assert est.n_refits == est.n_interior + est.n_clamped_high + est.n_clamped_low + (
        est.n_insufficient
    )
    assert 0.0 <= est.clamp_rate <= 1.0


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
