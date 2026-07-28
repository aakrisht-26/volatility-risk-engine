"""Christoffersen (1998) independence and conditional-coverage tests.

Kupiec's POF test asks whether breaches happen at the RIGHT RATE. It cannot
see clustering: 88 breaches spread evenly and 88 breaches all inside one
crisis month score identically. Christoffersen's independence test asks the
complementary question — given a breach yesterday, is a breach today more
likely? — by fitting a first-order Markov chain to the breach indicator.

Notation, over the ordered breach series I_1..I_n:

    n_ij = #{t : I_{t-1} = i, I_t = j}
    pi_01 = n_01 / (n_00 + n_01)      P(breach | no breach yesterday)
    pi_11 = n_11 / (n_10 + n_11)      P(breach | breach yesterday)
    pi    = (n_01 + n_11) / n_transitions

    LR_ind = -2 [ l(pi) - l(pi_01, pi_11) ]   ~ chi-square(1) under H0

H0 is pi_01 == pi_11 (independence). The test says nothing about the LEVEL of
the breach rate, which is why it is reported beside Kupiec rather than instead
of it. Christoffersen's conditional-coverage test combines both:

    LR_cc = LR_uc + LR_ind ~ chi-square(2)

rejecting when the breach process has the wrong rate, the wrong clustering, or
both. All log-likelihood terms use the 0*log(0) = 0 limit, which is what makes
the common degenerate cases (no consecutive breaches; no breaches at all)
well-defined rather than NaN.

Power caveat, same as Kupiec's: at 99% with n ~ 1,760 the expected breach count
is ~17.6, so n_11 is expected to be near zero even under clustering. A
non-rejection at 99% is weak evidence of independence, not proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class ChristoffersenResult:
    n_00: int
    n_01: int
    n_10: int
    n_11: int
    lr_ind: float
    p_ind: float

    @property
    def pi_01(self) -> float:
        """P(breach today | no breach yesterday); NaN when undefined."""
        denom = self.n_00 + self.n_01
        return self.n_01 / denom if denom else math.nan

    @property
    def pi_11(self) -> float:
        """P(breach today | breach yesterday); NaN when undefined."""
        denom = self.n_10 + self.n_11
        return self.n_11 / denom if denom else math.nan


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with the 0 * log(0) = 0 limit."""
    if x == 0:
        return 0.0
    return x * math.log(y)


def transition_counts(breaches: np.ndarray) -> tuple[int, int, int, int]:
    """(n_00, n_01, n_10, n_11) over consecutive pairs of the breach series."""
    b = np.asarray(breaches).astype(bool)
    if b.size < 2:
        return 0, 0, 0, 0
    prev, curr = b[:-1], b[1:]
    return (
        int(np.sum(~prev & ~curr)),
        int(np.sum(~prev & curr)),
        int(np.sum(prev & ~curr)),
        int(np.sum(prev & curr)),
    )


def christoffersen_independence(breaches: np.ndarray) -> ChristoffersenResult:
    """Markov-chain independence test on an ordered 0/1 breach series."""
    n_00, n_01, n_10, n_11 = transition_counts(breaches)
    n_trans = n_00 + n_01 + n_10 + n_11
    if n_trans == 0:
        return ChristoffersenResult(0, 0, 0, 0, 0.0, 1.0)

    pi = (n_01 + n_11) / n_trans
    ll_restricted = _xlogy(n_00 + n_10, 1 - pi) + _xlogy(n_01 + n_11, pi)

    denom_0, denom_1 = n_00 + n_01, n_10 + n_11
    pi_01 = n_01 / denom_0 if denom_0 else 0.0
    pi_11 = n_11 / denom_1 if denom_1 else 0.0
    ll_unrestricted = (
        _xlogy(n_00, 1 - pi_01)
        + _xlogy(n_01, pi_01)
        + _xlogy(n_10, 1 - pi_11)
        + _xlogy(n_11, pi_11)
    )

    lr = max(-2.0 * (ll_restricted - ll_unrestricted), 0.0)
    return ChristoffersenResult(n_00, n_01, n_10, n_11, lr, float(chi2.sf(lr, df=1)))


def conditional_coverage(lr_uc: float, lr_ind: float) -> tuple[float, float]:
    """Christoffersen's joint test: (LR_cc, p) with LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    lr_cc = lr_uc + lr_ind
    return lr_cc, float(chi2.sf(lr_cc, df=2))
