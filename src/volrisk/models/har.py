"""HAR-RV: linear regression of next-day variance on daily/weekly/monthly RV.

Corsi's heterogeneous-autoregressive structure on the Garman-Klass variance
proxy: the daily component IS gk_var; the weekly and monthly components are
its 5- and 22-day means, precomputed in the features layer. Fitted by OLS
(scikit-learn) inside the shared walk-forward harness.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

HAR_FEATURES: tuple[str, ...] = ("gk_var", "har_rv_w", "har_rv_m")


def make_har_model() -> LinearRegression:
    return LinearRegression()


def har_coefficients(model: LinearRegression) -> dict[str, float]:
    """Named OLS coefficients of a fitted HAR model (for reporting)."""
    out = {"const": float(model.intercept_)}
    out.update({name: float(c) for name, c in zip(HAR_FEATURES, model.coef_, strict=True)})
    return out
