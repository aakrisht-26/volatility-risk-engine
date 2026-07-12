"""Forecast-evaluation losses.

QLIKE is primary (CLAUDE.md). Exact parameterization used here, with h the
realized variance and f the forecast (both daily variances in return units):

    QLIKE(h, f) = h/f - ln(h/f) - 1

This is the Patton (2011)-class robust loss normalized so a perfect forecast
scores exactly 0; other texts write the un-normalized ln(f) + h/f, which has
the same minimizer but a data-dependent minimum. QLIKE is dimensionless,
lower is better, and deliberately asymmetric: it punishes variance
UNDER-forecasts far harder than over-forecasts — the right asymmetry for
risk work.

RMSE is secondary, computed in ANNUALIZED-VOLATILITY percentage points so the
number is readable ("vol points"): rmse(100*sqrt(252*h), 100*sqrt(252*f)).
Inputs everywhere are daily variances in return units, same as
features.gk_var and forecasts.daily_variance.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """Mean of h/f - ln(h/f) - 1. Requires strictly positive inputs."""
    h = np.asarray(realized_var, dtype=float)
    f = np.asarray(forecast_var, dtype=float)
    ratio = h / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse_ann_vol_pct(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """RMSE between annualized volatilities, in percentage points."""
    a = np.sqrt(TRADING_DAYS * np.asarray(realized_var, dtype=float)) * 100.0
    b = np.sqrt(TRADING_DAYS * np.asarray(forecast_var, dtype=float)) * 100.0
    return float(np.sqrt(np.mean((a - b) ** 2)))
