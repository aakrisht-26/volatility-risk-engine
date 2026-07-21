"""GARCH(1,1) walk-forward next-day variance baseline, fitted via arch.

Scaling contract (recorded Step-7 requirement): returns are scaled x100
(percent) before fitting because arch's optimizer is numerically happier at
that scale — and every forecast is converted back (÷10 000), so everything
this module emits is DAILY variance in RETURN units, consistent with
features.gk_var and forecasts.daily_variance. Nothing percent-scaled ever
leaves this module.

Convergence policy (recorded Step-8 addendum): a refit that raises or reports
non-convergence is never silently consumed and never crashes the walk-forward.
It logs a WARNING with the refit date and falls back to the previous refit's
parameters; the counts are returned so the CLI can print a canary-style
convergence summary. The only unrecoverable case is the FIRST fit raising —
there is nothing to fall back to. A first fit that merely fails to converge is
consumed loudly (logged as ERROR, counted separately).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

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


@dataclass(frozen=True)
class GarchFit:
    """Outcome of one MLE attempt. ``params`` is None when the optimizer raised."""

    params: GarchParams | None
    converged: bool


@dataclass
class GarchWalkForwardResult:
    forecasts: pd.Series
    refits: int = 0
    fallback_refits: int = 0  # failed refits where the previous refit's params were reused
    unconverged_consumed: int = 0  # first fit unconverged, consumed loudly (no fallback exists)
    fallback_dates: list[date] = field(default_factory=list)
    final_params: GarchParams | None = None  # the last refit's params (for the live row)


def garch_next_step(params: GarchParams, last_return: float, last_variance: float) -> float:
    """One GARCH step in RETURN units: the live next-session variance.

    ``last_return`` is the most recent completed session's log return and
    ``last_variance`` that session's conditional variance forecast (return
    units, as stored). Scaling to arch's percent space and back is internal.
    """
    eps_pct = last_return * RETURN_TO_PCT - params.mu
    s2_pct = last_variance / PCT_VAR_TO_RETURN_VAR
    live_pct = params.omega + params.alpha * eps_pct**2 + params.beta * s2_pct
    return live_pct * PCT_VAR_TO_RETURN_VAR


def fit_garch_params(returns_pct: pd.Series) -> GarchFit:
    """One MLE fit of a constant-mean GARCH(1,1) on percent-scaled returns.

    Never raises: optimizer exceptions and non-convergence are reported through
    the returned :class:`GarchFit` so the walk-forward can apply its policy.
    """
    am = arch_model(
        returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    try:
        res = am.fit(disp="off", show_warning=False)
    except Exception as exc:
        logger.warning("GARCH optimizer raised (n=%d): %s", len(returns_pct), exc)
        return GarchFit(params=None, converged=False)
    p = res.params
    params = GarchParams(
        float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
    )
    return GarchFit(params=params, converged=res.convergence_flag == 0)


def filter_conditional_variance(returns_pct: np.ndarray, params: GarchParams) -> np.ndarray:
    """``s2[t]`` = conditional variance OF session t given info through t-1 (pct² units).

    Seeded with the unconditional variance omega/(1-alpha-beta) (sample
    variance if the process is non-stationary).
    """
    eps = returns_pct - params.mu
    n = len(eps)
    s2 = np.empty(n)
    persistence = params.alpha + params.beta
    s2[0] = params.omega / (1.0 - persistence) if persistence < 1.0 else float(np.var(eps))
    for t in range(1, n):
        s2[t] = params.omega + params.alpha * eps[t - 1] ** 2 + params.beta * s2[t - 1]
    return s2


def garch_variance_forecasts(returns: pd.Series, min_train: int = 756) -> GarchWalkForwardResult:
    """Walk-forward next-day variance forecasts, indexed by the session forecast.

    Expanding window; refit at the first session of each calendar month
    (CLAUDE.md: minimum ~3y train, refit monthly). Between refits the GARCH
    recursion steps forward daily with fixed parameters. The fit used for the
    forecast of session t sees returns through t-1 only. Failed refits follow
    the module-docstring convergence policy.
    """
    if len(returns) <= min_train:
        return GarchWalkForwardResult(forecasts=pd.Series(dtype=float))
    dates = list(returns.index)
    r_pct = returns.to_numpy(dtype=float) * RETURN_TO_PCT
    n = len(r_pct)
    out = np.full(n, np.nan)
    result = GarchWalkForwardResult(forecasts=pd.Series(dtype=float))

    params: GarchParams | None = None
    refit_month: tuple[int, int] | None = None
    s2_prev = np.nan  # conditional variance OF session t-1, pct²
    eps_prev = np.nan

    for t in range(min_train, n):
        month = (dates[t].year, dates[t].month)
        if params is None or month != refit_month:
            fit = fit_garch_params(pd.Series(r_pct[:t]))
            result.refits += 1
            if fit.converged and fit.params is not None:
                new_params = fit.params
            elif params is not None:
                logger.warning(
                    "GARCH refit failed at %s (converged=%s); falling back to previous "
                    "refit's parameters",
                    dates[t],
                    fit.converged,
                )
                result.fallback_refits += 1
                result.fallback_dates.append(dates[t])
                new_params = params
            elif fit.params is not None:
                logger.error(
                    "first GARCH fit did not converge at %s; consuming unconverged "
                    "parameters loudly (no fallback exists)",
                    dates[t],
                )
                result.unconverged_consumed += 1
                new_params = fit.params
            else:
                raise RuntimeError(
                    f"first GARCH fit raised at {dates[t]}; no fallback parameters exist"
                )
            if new_params is not params:
                # Parameters changed: re-filter history to rebuild the recursion state.
                # On fallback the params are identical, so the daily-stepped state is
                # already exact and re-filtering is skipped.
                s2_prev = filter_conditional_variance(r_pct[:t], new_params)[-1]
                eps_prev = r_pct[t - 1] - new_params.mu
            params = new_params
            refit_month = month
        out[t] = params.omega + params.alpha * eps_prev**2 + params.beta * s2_prev
        s2_prev = out[t]
        eps_prev = r_pct[t] - params.mu

    result.forecasts = (pd.Series(out, index=returns.index) * PCT_VAR_TO_RETURN_VAR).iloc[
        min_train:
    ]
    result.final_params = params
    return result
