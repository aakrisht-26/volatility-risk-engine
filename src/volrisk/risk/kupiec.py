"""Kupiec (1995) proportion-of-failures (POF) unconditional-coverage test.

Under H0 the breach indicator is i.i.d. Bernoulli(p) with p the VaR tail
probability (1 - alpha). With n observations and x breaches the likelihood-ratio
statistic is

    LR_POF = -2 * [ x*ln(p) + (n-x)*ln(1-p) - x*ln(x/n) - (n-x)*ln(1 - x/n) ],

asymptotically chi-square(1) under H0. A small p-value rejects correct coverage:
too many breaches (VaR too small) or too few (VaR too conservative).

Power caveat (relevant at 99%): with n ~ 1,756 the expected breach count is only
~17.6 at 99%, so the test has low power there — a non-rejection is weak evidence
of correct coverage, not proof of it. Interpret 99% p-values accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import chi2


@dataclass(frozen=True)
class KupiecResult:
    n_obs: int
    n_breaches: int
    tail_prob: float
    lr_stat: float
    p_value: float


def kupiec_pof(n_obs: int, n_breaches: int, tail_prob: float) -> KupiecResult:
    """Kupiec POF test. ``tail_prob`` is the expected breach probability (1 - alpha)."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if not 0 < tail_prob < 1:
        raise ValueError("tail_prob must be in (0, 1)")
    x, n, p = n_breaches, n_obs, tail_prob

    # Restricted log-likelihood (breach prob fixed at p).
    ll_restricted = x * math.log(p) + (n - x) * math.log(1 - p)
    # Unrestricted log-likelihood (breach prob = x/n), with the 0*log(0) = 0 limit.
    pi_hat = x / n
    ll_unrestricted = 0.0
    if x > 0:
        ll_unrestricted += x * math.log(pi_hat)
    if x < n:
        ll_unrestricted += (n - x) * math.log(1 - pi_hat)

    lr = -2.0 * (ll_restricted - ll_unrestricted)
    lr = max(lr, 0.0)  # guard against tiny negative floating-point noise at pi_hat == p
    p_value = float(chi2.sf(lr, df=1))
    return KupiecResult(n_obs=n, n_breaches=x, tail_prob=p, lr_stat=lr, p_value=p_value)
