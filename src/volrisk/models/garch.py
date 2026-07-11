"""GARCH(1,1) walk-forward next-day variance baseline, fitted via arch.

Scaling contract (recorded Step-7 requirement): returns are scaled x100
(percent) before fitting because arch's optimizer is numerically happier at
that scale — and every forecast is converted back (÷10 000), so everything
this module emits is DAILY variance in RETURN units, consistent with
features.gk_var and forecasts.daily_variance. Nothing percent-scaled ever
leaves this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

RETURN_TO_PCT = 100.0
PCT_VAR_TO_RETURN_VAR = 1.0 / (RETURN_TO_PCT**2)


@dataclass(frozen=True)
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float


def fit_garch_params(returns_pct: pd.Series) -> GarchParams:
    """MLE fit of a constant-mean GARCH(1,1) on percent-scaled returns."""
    am = arch_model(
        returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    res = am.fit(disp="off", show_warning=False)
    if res.convergence_flag != 0:
        logger.warning(
            "GARCH fit did not converge (flag=%d, n=%d)", res.convergence_flag, len(returns_pct)
        )
    p = res.params
    return GarchParams(float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"]))


def filter_conditional_variance(returns_pct: np.ndarray, params: GarchParams) -> np.ndarray:
    """``s2[t]`` = conditional variance OF session t given info through t-1 (pct² units).

    Seeded with the unconditional variance omega/(1-alpha-beta) (sample variance if the
    process is non-stationary).
    """
    eps = returns_pct - params.mu
    n = len(eps)
    s2 = np.empty(n)
    persistence = params.alpha + params.beta
    s2[0] = params.omega / (1.0 - persistence) if persistence < 1.0 else float(np.var(eps))
    for t in range(1, n):
        s2[t] = params.omega + params.alpha * eps[t - 1] ** 2 + params.beta * s2[t - 1]
    return s2


def garch_variance_forecasts(returns: pd.Series, min_train: int = 756) -> pd.Series:
    """Walk-forward next-day variance forecasts, indexed by the session forecast.

    Expanding window; refit at the first session of each calendar month
    (CLAUDE.md: minimum ~3y train, refit monthly). Between refits the GARCH
    recursion steps forward daily with fixed parameters. The fit used for the
    forecast of session t sees returns through t-1 only.
    """
    if len(returns) <= min_train:
        return pd.Series(dtype=float)
    dates = list(returns.index)
    r_pct = returns.to_numpy(dtype=float) * RETURN_TO_PCT
    n = len(r_pct)
    out = np.full(n, np.nan)

    params: GarchParams | None = None
    refit_month: tuple[int, int] | None = None
    s2_prev = np.nan  # conditional variance OF session t-1, pct²
    eps_prev = np.nan

    for t in range(min_train, n):
        month = (dates[t].year, dates[t].month)
        if params is None or month != refit_month:
            params = fit_garch_params(pd.Series(r_pct[:t]))
            s2_prev = filter_conditional_variance(r_pct[:t], params)[-1]
            eps_prev = r_pct[t - 1] - params.mu
            refit_month = month
            logger.debug("refit at %s: %s", dates[t], params)
        out[t] = params.omega + params.alpha * eps_prev**2 + params.beta * s2_prev
        s2_prev = out[t]
        eps_prev = r_pct[t] - params.mu

    return (pd.Series(out, index=returns.index) * PCT_VAR_TO_RETURN_VAR).iloc[min_train:]
