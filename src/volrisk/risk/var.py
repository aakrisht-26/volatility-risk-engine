"""1-day parametric Value-at-Risk from a variance forecast.

Model: a zero-mean normal 1-day return. Over one trading day the drift term is
negligible next to volatility (mu ~ 1e-4 vs sigma ~ 1e-2), so it is dropped;
this is the standard 1-day parametric-VaR assumption and is stated in the
README. VaR is reported as a positive loss magnitude in log-return units:

    VaR_alpha(d) = z_alpha * sqrt(var_forecast(d))

with z_alpha the standard-normal quantile (z_95 = 1.645, z_99 = 2.326) and
var_forecast the model's daily variance in RETURN units, so sigma and VaR are
directly comparable to the realized close-to-close log return.

Breach convention (the classic off-by-sign trap): session d breaches when the
realized return r_d < -VaR_alpha(d) — i.e. the long-position loss exceeded the
VaR. P(r_d < -z_alpha * sigma) = 1 - alpha under the model, so the expected
breach rate is exactly 1 - alpha (5% at level 95, 1% at level 99).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

#: VaR confidence levels and their derived constants.
LEVELS: tuple[int, ...] = (95, 99)
Z_SCORE: dict[int, float] = {level: float(norm.ppf(level / 100.0)) for level in LEVELS}
TAIL_PROB: dict[int, float] = {level: 1.0 - level / 100.0 for level in LEVELS}


def var_threshold(variance: np.ndarray | pd.Series, level: int) -> np.ndarray:
    """Positive VaR loss magnitude z_alpha * sqrt(variance) for each forecast."""
    v = np.asarray(variance, dtype=float)
    return Z_SCORE[level] * np.sqrt(v)


def breach_mask(log_return: np.ndarray | pd.Series, threshold: np.ndarray) -> np.ndarray:
    """Boolean breach indicator: realized return below the negative VaR threshold."""
    return np.asarray(log_return, dtype=float) < -np.asarray(threshold, dtype=float)
