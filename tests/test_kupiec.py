"""Tests for the Kupiec POF test — hand values and the edge cases that bite."""

import math

import pytest
from scipy.stats import chi2

from volrisk.risk.kupiec import kupiec_pof


def test_perfect_coverage_gives_zero_statistic() -> None:
    # x/n == p exactly: restricted and unrestricted likelihoods coincide.
    result = kupiec_pof(n_obs=1000, n_breaches=50, tail_prob=0.05)
    assert result.lr_stat == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


def test_matches_closed_form_hand_value() -> None:
    n, x, p = 1000, 100, 0.05
    result = kupiec_pof(n, x, p)

    pi = x / n
    expected_lr = -2 * (
        x * math.log(p) + (n - x) * math.log(1 - p) - x * math.log(pi) - (n - x) * math.log(1 - pi)
    )
    assert result.lr_stat == pytest.approx(expected_lr, rel=1e-12)
    assert result.p_value == pytest.approx(float(chi2.sf(expected_lr, 1)), rel=1e-12)


def test_zero_breaches_uses_the_log_zero_limit() -> None:
    # x=0 is the common 99% over-coverage case; x*ln(x/n) must resolve to 0.
    result = kupiec_pof(n_obs=1756, n_breaches=0, tail_prob=0.01)
    assert result.lr_stat == pytest.approx(-2 * 1756 * math.log(0.99), rel=1e-12)
    assert result.p_value < 1e-6  # zero breaches at 99% is significant over-coverage


def test_all_breaches_uses_the_other_limit() -> None:
    result = kupiec_pof(n_obs=100, n_breaches=100, tail_prob=0.05)
    assert result.lr_stat == pytest.approx(-2 * 100 * math.log(0.05), rel=1e-12)


def test_more_breaches_raises_the_statistic_monotonically() -> None:
    p = 0.05
    lrs = [kupiec_pof(1000, x, p).lr_stat for x in (50, 70, 100, 150)]
    assert lrs == sorted(lrs)  # further from nominal -> larger LR


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        kupiec_pof(0, 0, 0.05)
    with pytest.raises(ValueError):
        kupiec_pof(100, 5, 0.0)
    with pytest.raises(ValueError):
        kupiec_pof(100, 5, 1.0)
