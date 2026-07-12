"""Log-variance target wrapper for level-space regressors (ablation v2 ruling).

Why log space: v1 fit variance in LEVELS, and both OLS and gradient-boosted
trees emitted a handful of near-zero (HAR: even negative, floored) forecasts
in calm regimes. QLIKE — asymmetric by design, punishing variance
under-forecasts hardest, which is the right asymmetry for risk work — blew up
on exactly those dates. Fitting ln(variance) makes positivity structural
instead of floor-enforced.

Retransformation (explicit, per the ruling): lognormal half-variance
correction, exp(m + s2/2), where s2 is the training-residual variance in log
space, re-estimated at every refit. Raw exponentiation exp(m) estimates the
conditional MEDIAN under lognormal errors and systematically under-forecasts
the conditional mean — precisely the direction QLIKE and risk applications
punish. The correction assumes approximately Gaussian log-space residuals;
that assumption is documented and accepted rather than hidden.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LogVarianceRegressor:
    """Fit any level-space regressor on ln(y); predict variance in levels.

    Exposes the wrapped estimator as ``inner`` (e.g. for HAR coefficient
    reporting — those coefficients live in log-variance space).
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self._half_resid_var = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogVarianceRegressor:
        positive = np.asarray(y, dtype=float) > 0.0
        if not positive.all():
            # gk_var is provably non-negative on valid OHLC but can be exactly
            # zero on a degenerate flat bar; such targets carry no log-space
            # information and are dropped loudly.
            logger.warning(
                "dropping %d non-positive target(s) before log-space fit", int((~positive).sum())
            )
            X = X.loc[positive]
            y = y.loc[positive]
        log_y = np.log(np.asarray(y, dtype=float))
        self.inner.fit(X, log_y)
        resid = log_y - np.asarray(self.inner.predict(X), dtype=float)
        self._half_resid_var = 0.5 * float(np.var(resid))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(np.asarray(self.inner.predict(X), dtype=float) + self._half_resid_var)
