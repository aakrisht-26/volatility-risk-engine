"""Student-t VaR: fat-tailed quantiles scaled to the model's variance forecast.

The normal-quantile VaR under-covers at 99% for every model (measured, see the
README). The diagnosed cause is distributional, not a variance-level error:
a Gaussian quantile cannot reach the tail of real daily equity returns. This
module replaces the quantile while keeping each model's variance forecast
untouched, so the comparison isolates tail shape.

Scaling contract. A standard t with ``df`` degrees of freedom has variance
df/(df-2), not 1. To make the VaR distribution's variance equal the model's
variance forecast sigma^2, the quantile is rescaled:

    VaR_alpha = sigma * t_ppf(alpha, df) / sqrt(df / (df - 2))

so that only the SHAPE changes, never the variance. Requires df > 2 (below
that a t has no finite variance and "match the variance" is meaningless).

Estimator choice: **MLE on standardized residuals** z_t = r_t / sigma_t, with
location fixed at zero and scale free (only df is used downstream).
Moment-matching on excess kurtosis — df = 4 + 6/kurtosis — was rejected: it is
a fourth-moment statistic, so on exactly the fat-tailed data being modelled it
is dominated by a handful of observations and has enormous sampling variance.
Worse, for df <= 4 the population kurtosis does not exist while the sample
version still returns a finite number, so moment matching fails silently
precisely where fat tails matter most. MLE uses the whole likelihood, is
consistent, and its failure mode is a boundary hit that can be detected and
logged rather than a plausible-looking wrong answer.

Estimation is walk-forward: df is re-estimated at each monthly refit boundary
from training data strictly before that month, exactly like the variance
calibration factors, so no VaR threshold sees its own outcome.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

#: df must exceed 2 for a finite variance; 100 is numerically indistinguishable
#: from the normal for quantile purposes. Hits on either bound are logged.
MIN_DF = 2.05
MAX_DF = 100.0

#: Minimum standardized residuals before an MLE fit is attempted.
MIN_FIT_OBS = 100


def t_quantile_scaled(df: float, alpha: float) -> float:
    """Variance-matched t quantile: t_ppf(alpha, df) / sqrt(df/(df-2)).

    Returns the multiplier applied to sigma, directly comparable to the normal
    z-score. Note it is SMALLER than the normal z at 95% and LARGER at 99% —
    a variance-matched t trades shoulder mass for tail mass.
    """
    if df <= 2:
        raise ValueError(f"df must exceed 2 for a finite variance, got {df}")
    return float(stats.t.ppf(alpha, df) / np.sqrt(df / (df - 2.0)))


def fit_t_df(standardized_residuals: np.ndarray) -> float:
    """MLE degrees of freedom for standardized residuals (location fixed at 0)."""
    z = np.asarray(standardized_residuals, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < MIN_FIT_OBS:
        logger.warning("only %d observations for the t fit; falling back to df=%s", z.size, MAX_DF)
        return MAX_DF
    df, _loc, _scale = stats.t.fit(z, floc=0.0)
    clamped = float(np.clip(df, MIN_DF, MAX_DF))
    if clamped != df:
        logger.warning("t df %.3f clamped to %.3f (bounds %s..%s)", df, clamped, MIN_DF, MAX_DF)
    return clamped


def walk_forward_t_df(
    returns: pd.Series, variance: pd.Series, min_train: int = MIN_FIT_OBS
) -> pd.Series:
    """Per-date degrees of freedom, re-estimated at each calendar-month boundary.

    ``returns`` and ``variance`` share an index of forecast dates. The df in
    force for month M is fitted on standardized residuals strictly BEFORE M's
    first date, so a threshold never sees the return it is tested against.
    Dates before ``min_train`` observations exist use the normal-equivalent
    MAX_DF (a deliberate no-op rather than a guess).
    """
    z_all = (returns / np.sqrt(variance)).to_numpy(dtype=float)
    dates = list(returns.index)

    out: dict[date, float] = {}
    month: tuple[int, int] | None = None
    current = MAX_DF
    for i, d in enumerate(dates):
        m = (d.year, d.month)
        if m != month:
            current = fit_t_df(z_all[:i]) if i >= min_train else MAX_DF
            month = m
        out[d] = current
    return pd.Series(out, dtype=float)


def t_var_thresholds(variance: pd.Series, df_series: pd.Series, alpha: float) -> np.ndarray:
    """Positive VaR magnitudes using each date's variance-matched t quantile."""
    sigma = np.sqrt(np.asarray(variance, dtype=float))
    quantiles = np.array([t_quantile_scaled(float(d), alpha) for d in df_series])
    return sigma * quantiles
