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
from dataclasses import dataclass, field
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


#: Estimation outcomes. A high clamp means the MLE ran to the Gaussian boundary
#: (no fat tails detectable in that training window), so the t variant is
#: effectively the normal one there. A low clamp means the MLE wanted df <= 2,
#: where a t has no finite variance and the variance-matched quantile COLLAPSES
#: (at df = 2.05 the 99% multiplier is 1.05 against the normal's 2.33) — that is
#: a silently dangerous VaR, which is why both are counted, not just logged.
INTERIOR = "interior"
CLAMPED_HIGH = "clamped_high"
CLAMPED_LOW = "clamped_low"
INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class TDfFit:
    raw_df: float
    df: float
    status: str


@dataclass
class TDfEstimation:
    """Per-date df plus the refit-level diagnostics behind it."""

    df_series: pd.Series
    n_refits: int = 0
    n_interior: int = 0
    n_clamped_high: int = 0
    n_clamped_low: int = 0
    n_insufficient: int = 0
    interior_dfs: list[float] = field(default_factory=list)
    clamp_dates: list[tuple[date, str, float]] = field(default_factory=list)

    @property
    def clamp_rate(self) -> float:
        """Share of refits that hit either bound (insufficient-data refits excluded)."""
        estimated = self.n_refits - self.n_insufficient
        return (self.n_clamped_high + self.n_clamped_low) / estimated if estimated else 0.0

    @property
    def interior_median(self) -> float:
        return float(np.median(self.interior_dfs)) if self.interior_dfs else float("nan")


def fit_t_df(standardized_residuals: np.ndarray) -> TDfFit:
    """MLE degrees of freedom for standardized residuals (location fixed at 0).

    Returns the raw estimate alongside the clamped value and a status, so a
    boundary hit is a reportable event rather than an invisible rescue.
    """
    z = np.asarray(standardized_residuals, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < MIN_FIT_OBS:
        logger.warning("only %d observations for the t fit; falling back to df=%s", z.size, MAX_DF)
        return TDfFit(raw_df=float("nan"), df=MAX_DF, status=INSUFFICIENT)

    raw, _loc, _scale = stats.t.fit(z, floc=0.0)
    raw = float(raw)
    if raw >= MAX_DF:
        return TDfFit(raw, MAX_DF, CLAMPED_HIGH)
    if raw <= MIN_DF:
        return TDfFit(raw, MIN_DF, CLAMPED_LOW)
    return TDfFit(raw, raw, INTERIOR)


def walk_forward_t_df(
    returns: pd.Series, variance: pd.Series, min_train: int = MIN_FIT_OBS
) -> TDfEstimation:
    """Per-date degrees of freedom, re-estimated at each calendar-month boundary.

    ``returns`` and ``variance`` share an index of forecast dates. The df in
    force for month M is fitted on standardized residuals strictly BEFORE M's
    first date, so a threshold never sees the return it is tested against.
    Dates before ``min_train`` observations exist use the normal-equivalent
    MAX_DF (a deliberate no-op rather than a guess).

    Returns the series together with refit-level clamp diagnostics: a high
    clamp means the t is indistinguishable from the normal for that window, so
    the counts are what tell you whether the t variant did any work at all.
    """
    z_all = (returns / np.sqrt(variance)).to_numpy(dtype=float)
    dates = list(returns.index)

    out: dict[date, float] = {}
    result = TDfEstimation(df_series=pd.Series(dtype=float))
    month: tuple[int, int] | None = None
    current = MAX_DF
    for i, d in enumerate(dates):
        m = (d.year, d.month)
        if m != month:
            fit = (
                fit_t_df(z_all[:i])
                if i >= min_train
                else TDfFit(float("nan"), MAX_DF, INSUFFICIENT)
            )
            # A LOW clamp is a failed fit, not an estimate: consuming it would
            # emit a VaR far SMALLER than the Gaussian one. Same remedy as the
            # GARCH convergence policy — never consume a failed fit; fall back
            # to the last good df, or to the normal-equivalent if none exists.
            current = current if fit.status == CLAMPED_LOW else fit.df
            result.n_refits += 1
            if fit.status == INTERIOR:
                result.n_interior += 1
                result.interior_dfs.append(fit.df)
            elif fit.status == CLAMPED_HIGH:
                result.n_clamped_high += 1
                result.clamp_dates.append((d, CLAMPED_HIGH, fit.raw_df))
                logger.warning("t df %.3g at %s clamped HIGH to %.2f", fit.raw_df, d, MAX_DF)
            elif fit.status == CLAMPED_LOW:
                result.n_clamped_low += 1
                result.clamp_dates.append((d, CLAMPED_LOW, fit.raw_df))
                logger.error(
                    "t df %.3f at %s is below %.2f (no finite variance): fit REJECTED, "
                    "falling back to df=%.2f — consuming it would collapse the "
                    "variance-matched quantile below the Gaussian one",
                    fit.raw_df,
                    d,
                    MIN_DF,
                    current,
                )
            else:
                result.n_insufficient += 1
            month = m
        out[d] = current
    result.df_series = pd.Series(out, dtype=float)
    return result


def t_var_thresholds(variance: pd.Series, df_series: pd.Series, alpha: float) -> np.ndarray:
    """Positive VaR magnitudes using each date's variance-matched t quantile."""
    sigma = np.sqrt(np.asarray(variance, dtype=float))
    quantiles = np.array([t_quantile_scaled(float(d), alpha) for d in df_series])
    return sigma * quantiles
