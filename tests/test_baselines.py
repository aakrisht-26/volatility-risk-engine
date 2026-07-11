"""Tests for the EWMA and GARCH walk-forward baselines.

The two bug classes that matter here are off-by-one leakage (a forecast seeing
its own day's return) and unit mismatches (percent² leaking out of the arch
layer). Both get explicit tests: hand recursions pin the indexing, a
tail-mutation test pins no-leakage, and a magnitude test pins the x10 000
round-trip. Only the fit smoke test touches arch's optimizer; everything else
uses fixed parameters.
"""

import os
from datetime import date

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from sqlalchemy import text

import volrisk.models.garch as garch_module
from volrisk.db.loaders import upsert_variance_forecasts
from volrisk.models.ewma import ewma_variance_forecasts
from volrisk.models.garch import (
    GarchParams,
    filter_conditional_variance,
    fit_garch_params,
    garch_variance_forecasts,
)

requires_db = pytest.mark.skipif(
    not os.environ.get("VOLRISK_TEST_DATABASE_URL"), reason="VOLRISK_TEST_DATABASE_URL not set"
)

FIXED_PARAMS = GarchParams(mu=0.0, omega=0.02, alpha=0.05, beta=0.90)


def session_dates(n: int) -> list[date]:
    sessions = mcal.get_calendar("XNYS").schedule("2024-01-02", "2024-12-31").index[:n]
    return [ts.date() for ts in sessions]


def returns_series(n: int, seed: int = 11, scale: float = 0.02) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, scale, n), index=session_dates(n))


# --- EWMA ---


def test_ewma_matches_hand_recursion() -> None:
    r = returns_series(10)
    lam, min_train, seed_window = 0.94, 3, 2

    out = ewma_variance_forecasts(r, lam=lam, min_train=min_train, seed_window=seed_window)

    r2 = r.to_numpy() ** 2
    sigma2 = np.empty(len(r2))
    sigma2[0] = r2[:seed_window].mean()
    for t in range(1, len(r2)):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r2[t - 1]
    assert len(out) == 7
    np.testing.assert_allclose(out.to_numpy(), sigma2[min_train:], rtol=0, atol=0)


def test_ewma_forecast_uses_only_past_information() -> None:
    r = returns_series(40)
    out = ewma_variance_forecasts(r, min_train=10, seed_window=5)

    mutated = r.copy()
    mutated.iloc[25:] = 99.0  # corrupt the future
    out_mutated = ewma_variance_forecasts(mutated, min_train=10, seed_window=5)

    # Forecast FOR session 25 uses returns through 24 — unchanged. From 26 on, changed.
    pd.testing.assert_series_equal(out.loc[: r.index[25]], out_mutated.loc[: r.index[25]])
    assert (out_mutated.loc[r.index[26] :] != out.loc[r.index[26] :]).all()


def test_ewma_too_short_series_yields_empty() -> None:
    assert ewma_variance_forecasts(returns_series(20), min_train=20).empty


# --- GARCH filter and walk-forward (fixed parameters; no optimizer) ---


def test_garch_filter_matches_hand_recursion() -> None:
    eps = np.array([0.5, -1.2, 0.3, 2.0, -0.7])  # pct units, mu = 0

    s2 = filter_conditional_variance(eps, FIXED_PARAMS)

    expected = np.empty(5)
    expected[0] = FIXED_PARAMS.omega / (1 - FIXED_PARAMS.alpha - FIXED_PARAMS.beta)
    for t in range(1, 5):
        expected[t] = (
            FIXED_PARAMS.omega
            + FIXED_PARAMS.alpha * eps[t - 1] ** 2
            + FIXED_PARAMS.beta * expected[t - 1]
        )
    np.testing.assert_allclose(s2, expected, rtol=0, atol=0)


@pytest.fixture
def fixed_fit(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the MLE fit with fixed params; returns the list of fit-window lengths."""
    calls: list[int] = []

    def fake_fit(returns_pct: pd.Series) -> GarchParams:
        calls.append(len(returns_pct))
        return FIXED_PARAMS

    monkeypatch.setattr(garch_module, "fit_garch_params", fake_fit)
    return calls


def test_garch_refits_once_per_calendar_month(fixed_fit: list[int]) -> None:
    r = returns_series(40)  # ~2024-01-02 .. 2024-02-28: January + February

    garch_variance_forecasts(r, min_train=10)

    # One fit when forecasting starts (January), one at the first February session.
    assert len(fixed_fit) == 2
    assert fixed_fit[0] == 10  # expanding window: first fit sees exactly min_train returns


def test_garch_forecast_uses_only_past_information(fixed_fit: list[int]) -> None:
    r = returns_series(40)
    out = garch_variance_forecasts(r, min_train=10)

    mutated = r.copy()
    mutated.iloc[25:] = 0.5
    out_mutated = garch_variance_forecasts(mutated, min_train=10)

    pd.testing.assert_series_equal(out.loc[: r.index[25]], out_mutated.loc[: r.index[25]])
    assert (out_mutated.loc[r.index[26] :] != out.loc[r.index[26] :]).all()


def test_garch_output_is_daily_variance_in_return_units(fixed_fit: list[int]) -> None:
    r = returns_series(60, scale=0.02)
    out = garch_variance_forecasts(r, min_train=10)

    # With FIXED_PARAMS the unconditional pct² variance is 0.02/0.05 = 0.4,
    # i.e. 4e-5 in return units. A x10 000 mismatch would sit near 0.4.
    assert ((out > 1e-6) & (out < 1e-2)).all()


# --- arch integration (real optimizer, offline) ---


def test_fit_garch_params_smoke() -> None:
    rng = np.random.default_rng(5)
    r_pct = pd.Series(rng.normal(0.0, 1.2, 800))

    params = fit_garch_params(r_pct)

    assert params.omega > 0
    assert 0 <= params.alpha < 1
    assert 0 <= params.beta < 1
    assert params.alpha + params.beta < 1


# --- forecasts table integration ---


@requires_db
def test_forecast_upsert_is_idempotent_and_multi_model(db_engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE forecasts.daily_variance"))
    df = pd.DataFrame(
        {
            "ticker": "AAPL",
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "model": ["ewma_094", "garch_11"],
            "var_forecast": [1.2e-4, 1.5e-4],
        }
    )

    upsert_variance_forecasts(db_engine, df, context="test")
    upsert_variance_forecasts(db_engine, df, context="test")  # idempotent

    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM forecasts.daily_variance")).scalar_one()
    assert n == 2  # same (ticker, trade_date), two models — both kept, no duplicates
