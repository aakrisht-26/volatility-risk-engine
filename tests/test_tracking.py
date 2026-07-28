"""Tests for optional MLflow tracking.

The contract that matters is negative: with MLflow absent or disabled, every
entry point must no-op silently and the pipeline must be unaffected. Tracking
is observability; it may never fail a risk run.
"""

import builtins
from datetime import date

import pandas as pd
import pytest

from volrisk import tracking


def ablation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "^GSPC"],
            "model": ["har_rv", "har_rv"],
            "n_obs": [1762, 1762],
            "qlike": [0.30, 0.38],
            "rmse_ann_vol_pct": [9.41, 5.86],
            "eval_start": [date(2019, 7, 15)] * 2,
            "eval_end": [date(2026, 7, 17)] * 2,
        }
    )


def coverage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "^GSPC", "^GSPC"],
            "model": ["har_rv"] * 4,
            "level": [95, 99, 95, 99],
            "breach_rate": [0.077, 0.028, 0.094, 0.043],
            "kupiec_p": [0.001, 0.0001, 0.0, 0.0],
            "p_ind": [0.83, 0.74, 0.64, 0.69],
        }
    )


def test_caret_tickers_are_sanitized_for_mlflow_keys() -> None:
    assert tracking.sanitize_key("^GSPC") == "GSPC"
    assert tracking.sanitize_key("har_rv_cal_t") == "har_rv_cal_t"
    assert tracking.sanitize_key("RELIANCE.NS") == "RELIANCE.NS"


def test_disabled_by_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLRISK_DISABLE_MLFLOW", "1")

    assert tracking.is_enabled() is False
    assert tracking.log_evaluation(ablation_frame(), coverage_frame()) is None


def test_missing_mlflow_degrades_to_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate MLflow not being installed at all."""
    monkeypatch.delenv("VOLRISK_DISABLE_MLFLOW", raising=False)
    real_import = builtins.__import__

    def no_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("simulated: mlflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlflow)

    assert tracking.is_enabled() is False
    assert tracking.log_evaluation(ablation_frame(), coverage_frame()) is None


def test_mlflow_errors_are_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when MLflow is present, a logging failure must not propagate."""
    monkeypatch.delenv("VOLRISK_DISABLE_MLFLOW", raising=False)
    monkeypatch.setattr(tracking, "is_enabled", lambda: True)
    real_import = builtins.__import__

    def exploding_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise RuntimeError("simulated tracking-store failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", exploding_mlflow)

    assert tracking.log_evaluation(ablation_frame(), coverage_frame()) is None


def test_flatten_metrics_builds_sanitized_keys() -> None:
    flat = tracking._flatten_metrics(ablation_frame(), {"qlike": "qlike"})

    assert flat["qlike.har_rv.AAPL"] == pytest.approx(0.30)
    assert flat["qlike.har_rv.GSPC"] == pytest.approx(0.38)  # caret stripped


def test_ablation_markdown_contains_both_tables() -> None:
    md = tracking._ablation_markdown(ablation_frame())

    assert "## QLIKE" in md
    assert "## RMSE" in md
    assert "har_rv" in md
