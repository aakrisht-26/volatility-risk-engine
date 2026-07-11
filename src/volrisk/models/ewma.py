"""RiskMetrics EWMA(λ = 0.94) next-day variance baseline.

No parameters are estimated, so "walk-forward" is simply the recursion run
through time: the forecast FOR session t uses squared returns through t-1
only. Output values are DAILY variance in RETURN units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISKMETRICS_LAMBDA = 0.94
SEED_WINDOW = 252


def ewma_variance_forecasts(
    returns: pd.Series,
    lam: float = RISKMETRICS_LAMBDA,
    min_train: int = 756,
    seed_window: int = SEED_WINDOW,
) -> pd.Series:
    """Next-day variance forecasts, indexed by the session being forecast.

    ``sigma2[t] = lam·sigma2[t-1] + (1-lam)·r²[t-1]``, seeded with the mean
    squared return of the first ``seed_window`` observations. The seed peeks at
    that early window, but nothing is emitted before ``min_train`` sessions, by
    which point the seed's remaining weight is λ^(min_train - seed_window)
    ≈ 3e-14 — numerically extinct. Emitted forecasts therefore use information
    strictly through t-1.
    """
    if len(returns) <= min_train:
        return pd.Series(dtype=float)
    r2 = returns.to_numpy(dtype=float) ** 2
    n = len(r2)
    sigma2 = np.empty(n)
    sigma2[0] = r2[:seed_window].mean()
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * r2[t - 1]
    return pd.Series(sigma2, index=returns.index).iloc[min_train:]
