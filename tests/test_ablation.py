"""Integration tests for ablation metrics: computed from seeded DB rows,
hand-checked, stored idempotently, and rendered to markdown."""

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from volrisk.db.loaders import upsert_ablation_metrics, upsert_variance_forecasts
from volrisk.evaluate.ablation import compute_ablation, markdown_report

requires_db = pytest.mark.skipif(
    not os.environ.get("VOLRISK_TEST_DATABASE_URL"), reason="VOLRISK_TEST_DATABASE_URL not set"
)

DATES = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
REALIZED = 2.0e-4


@requires_db
def test_ablation_metrics_match_hand_computation(db_engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE forecasts.daily_variance"))
        conn.execute(text("TRUNCATE forecasts.ablation_metrics"))
        conn.execute(text("TRUNCATE features.daily_features"))
        for d in DATES:
            conn.execute(
                text(
                    "INSERT INTO features.daily_features (ticker, trade_date, gk_var)"
                    " VALUES ('SYN', :d, :gk)"
                ),
                {"d": d, "gk": REALIZED},
            )
    # Model A forecasts perfectly; model B forecasts half the realized variance.
    for model, factor in (("model_a", 1.0), ("model_b", 0.5)):
        upsert_variance_forecasts(
            db_engine,
            pd.DataFrame(
                {
                    "ticker": "SYN",
                    "trade_date": DATES,
                    "model": model,
                    "var_forecast": REALIZED * factor,
                }
            ),
            context="test",
        )

    metrics = compute_ablation(db_engine)

    assert len(metrics) == 2
    by_model = metrics.set_index("model")
    assert by_model.loc["model_a", "qlike"] == pytest.approx(0.0, abs=1e-15)
    # h/f = 2: qlike = 2 - ln 2 - 1
    assert by_model.loc["model_b", "qlike"] == pytest.approx(1.0 - np.log(2.0), rel=1e-12)
    assert by_model.loc["model_a", "n_obs"] == 3
    assert by_model.loc["model_a", "eval_start"] == DATES[0]
    assert by_model.loc["model_a", "eval_end"] == DATES[-1]

    n = upsert_ablation_metrics(db_engine, metrics)
    n_again = upsert_ablation_metrics(db_engine, metrics)  # idempotent
    assert n == n_again == 2
    with db_engine.connect() as conn:
        stored = conn.execute(text("SELECT count(*) FROM forecasts.ablation_metrics")).scalar_one()
    assert stored == 2


@requires_db
def test_markdown_report_contains_models_and_average(db_engine) -> None:
    metrics = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "model": ["ewma_094", "garch_11"] * 2,
            "n_obs": [10] * 4,
            "qlike": [0.5, 0.4, 0.6, 0.7],
            "rmse_ann_vol_pct": [5.0, 4.0, 6.0, 7.0],
            "eval_start": [date(2024, 1, 2)] * 4,
            "eval_end": [date(2024, 6, 28)] * 4,
        }
    )

    report = markdown_report(metrics)

    assert "ewma_094" in report and "garch_11" in report
    assert "**AVERAGE**" in report
    assert "QLIKE" in report and "RMSE" in report
    # Row-best bolding: AAA's best qlike is garch (0.4), BBB's is ewma (0.6).
    assert "**0.4000**" in report
    assert "**0.6000**" in report
