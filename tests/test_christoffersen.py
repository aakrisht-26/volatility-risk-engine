"""Tests for the Christoffersen independence and conditional-coverage tests.

The degenerate cases are the ones that bite: no consecutive breaches (n_11 = 0,
the normal case at 99%), no breaches at all, and all breaches. Each must
resolve through the 0*log(0) = 0 limit rather than producing NaN.
"""

import math

import numpy as np
import pytest
from scipy.stats import chi2

from volrisk.risk.christoffersen import (
    christoffersen_independence,
    conditional_coverage,
    transition_counts,
)


def test_transition_counts_by_hand() -> None:
    # series: 0 1 1 0 0 1  -> pairs (0,1),(1,1),(1,0),(0,0),(0,1)
    b = np.array([0, 1, 1, 0, 0, 1], dtype=bool)

    assert transition_counts(b) == (1, 2, 1, 1)


def test_short_series_has_no_transitions() -> None:
    assert transition_counts(np.array([1], dtype=bool)) == (0, 0, 0, 0)

    result = christoffersen_independence(np.array([1], dtype=bool))

    assert result.lr_ind == 0.0
    assert result.p_ind == 1.0


def test_perfectly_independent_series_scores_near_zero() -> None:
    # Alternating-free random series with a fixed rate and no memory.
    rng = np.random.default_rng(0)
    b = rng.random(20_000) < 0.05

    result = christoffersen_independence(b)

    assert result.lr_ind < 6.0  # chi2(1) 5% critical value is 3.84; allow sampling noise
    assert result.p_ind > 0.01


def test_clustered_series_is_rejected() -> None:
    # 1,760 sessions with the same 88 breaches (5%), but contiguous.
    b = np.zeros(1760, dtype=bool)
    b[400:488] = True

    result = christoffersen_independence(b)

    assert result.n_11 == 87  # 88 contiguous breaches -> 87 breach-after-breach pairs
    assert result.p_ind < 1e-6  # clustering is overwhelming
    assert result.pi_11 > result.pi_01


def test_perfectly_regular_series_is_also_rejected_negative_dependence() -> None:
    """The test is two-sided: TOO-regular breaches are non-independent too.

    88 breaches every 20th session gives n_11 = 0, but under independence at a
    5% rate you would expect ~4 breach-after-breach pairs. Deterministic spacing
    is negative dependence and is (correctly) rejected — the mirror image of
    clustering, and a reason to read n_11 alongside the p-value rather than
    treating rejection as synonymous with clustering.
    """
    b = np.zeros(1760, dtype=bool)
    b[::20] = True

    result = christoffersen_independence(b)

    assert result.n_11 == 0
    assert result.p_ind < 0.05
    assert result.pi_11 < result.pi_01  # rejected on the anti-clustering side


def test_no_breaches_is_degenerate_not_nan() -> None:
    result = christoffersen_independence(np.zeros(100, dtype=bool))

    assert (result.n_00, result.n_01, result.n_10, result.n_11) == (99, 0, 0, 0)
    assert result.lr_ind == 0.0
    assert result.p_ind == 1.0
    assert math.isnan(result.pi_11)  # undefined: never observed a breach day


def test_all_breaches_is_degenerate_not_nan() -> None:
    result = christoffersen_independence(np.ones(100, dtype=bool))

    assert (result.n_00, result.n_01, result.n_10, result.n_11) == (0, 0, 0, 99)
    assert result.lr_ind == 0.0
    assert math.isnan(result.pi_01)


def test_exactly_matching_conditional_rates_score_zero() -> None:
    # pairs give n_00=4, n_01=2, n_10=2, n_11=1: pi_01 = pi_11 = pi = 1/3 exactly,
    # so the statistic is identically 0 and must not drift negative on rounding.
    b = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)

    result = christoffersen_independence(b)

    assert (result.n_00, result.n_01, result.n_10, result.n_11) == (4, 2, 2, 1)
    assert result.lr_ind == 0.0
    assert result.p_ind == 1.0


def test_independence_matches_closed_form_hand_value() -> None:
    # Non-degenerate: pi_01 = 2/5, pi_11 = 2/4.
    b = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
    n_00, n_01, n_10, n_11 = transition_counts(b)
    n = n_00 + n_01 + n_10 + n_11
    pi = (n_01 + n_11) / n
    pi_01 = n_01 / (n_00 + n_01)
    pi_11 = n_11 / (n_10 + n_11)
    expected = -2 * (
        (n_00 + n_10) * math.log(1 - pi)
        + (n_01 + n_11) * math.log(pi)
        - n_00 * math.log(1 - pi_01)
        - n_01 * math.log(pi_01)
        - (n_10 * math.log(1 - pi_11) if pi_11 < 1 else 0.0)
        - (n_11 * math.log(pi_11) if n_11 else 0.0)
    )

    result = christoffersen_independence(b)

    assert result.lr_ind == pytest.approx(expected, rel=1e-12)
    assert result.p_ind == pytest.approx(float(chi2.sf(expected, 1)), rel=1e-12)


def test_conditional_coverage_sums_and_uses_two_df() -> None:
    lr_cc, p_cc = conditional_coverage(4.0, 2.0)

    assert lr_cc == 6.0
    assert p_cc == pytest.approx(float(chi2.sf(6.0, 2)), rel=1e-12)
    # Two degrees of freedom is more forgiving than one at the same statistic.
    assert p_cc > float(chi2.sf(6.0, 1))
