"""Forecast-evaluation losses.

QLIKE is primary (CLAUDE.md): robust to noise in the realized-variance proxy,
minimized at zero when forecast equals realized. RMSE is secondary and is
computed in ANNUALIZED-VOLATILITY percentage points so the number is readable
("vol points"); inputs everywhere are daily variances in return units, same as
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
