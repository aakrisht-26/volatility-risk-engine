"""Tests for optional MLflow tracking.

The contract that matters is negative: with MLflow absent or disabled, every
entry point must no-op silently and the pipeline must be unaffected. Tracking
is observability; it may never fail a risk run.

The positive path is covered too, but only when the ``tracking`` extra is
actually installed — CI runs the suite in both states so "MLflow present" is a
tested configuration rather than an assumed one.
"""

import builtins
import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volrisk import tracking

mlflow_installed = importlib.util.find_spec("mlflow") is not None
requires_mlflow = pytest.mark.skipif(
    not mlflow_installed, reason="optional 'tracking' extra not installed"
)


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


# --- the MLflow-present path (only meaningful with the extra installed) ---


@requires_mlflow
def test_real_mlflow_run_records_metrics_params_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end against a throwaway SQLite store in tmp_path.

    Everything else in this file simulates MLflow. This one uses it, so the
    parts that only fail against the real library — key sanitization rules, the
    SQLite backend, the create-then-set experiment dance — are exercised.
    """
    import mlflow

    monkeypatch.delenv("VOLRISK_DISABLE_MLFLOW", raising=False)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    run_id = tracking.log_evaluation(
        ablation_frame(),
        coverage_frame(),
        params={"refit_cadence": "calendar-monthly"},
        tracking_uri=uri,
        artifact_dir=tmp_path / "mlruns",
        run_name="pytest",
    )

    assert run_id is not None  # a None here means the wrapper swallowed a failure
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["refit_cadence"] == "calendar-monthly"
    assert run.data.metrics["qlike.har_rv.GSPC"] == pytest.approx(0.38)
    assert run.data.metrics["kupiec_rejects_99.har_rv"] == pytest.approx(2.0)
    assert {"ablation.csv", "var_coverage.csv", "ablation.md"} <= {
        f.path for f in mlflow.artifacts.list_artifacts(run_id=run_id)
    }


@requires_mlflow
def test_pandas_stayed_on_3x_with_the_tracking_extra_installed() -> None:
    """The standing pin tripwire, as an executable check.

    The full ``mlflow`` package pins pandas<3; ``mlflow-skinny`` does not. If a
    future lock swaps one for the other, this fails loudly here instead of
    silently downgrading the whole project's dataframe stack.
    """
    assert int(pd.__version__.split(".")[0]) >= 3, (
        f"the tracking extra pulled pandas back to {pd.__version__}; "
        "the standing pandas<3 tripwire has fired"
    )
