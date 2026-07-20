"""The dashboard schema (migration 009) must exist and every view must be
queryable — Power BI's whole surface is these six views."""

import os

import pytest
from sqlalchemy import text

requires_db = pytest.mark.skipif(
    not os.environ.get("VOLRISK_TEST_DATABASE_URL"), reason="VOLRISK_TEST_DATABASE_URL not set"
)

VIEWS = (
    "v_forecast_vs_realized",
    "v_var_daily",
    "v_var_coverage",
    "v_ablation",
    "v_vol_regime",
    "v_latest_forecast",
)


@requires_db
def test_every_dashboard_view_is_queryable(db_engine) -> None:
    with db_engine.connect() as conn:
        for view in VIEWS:
            conn.execute(text(f"SELECT * FROM dashboard.{view} LIMIT 1"))  # must not raise


@requires_db
def test_deferred_indexes_exist(db_engine) -> None:
    with db_engine.connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes"
                    " WHERE indexname IN ('daily_variance_model_date_idx',"
                    " 'clean_daily_bars_date_idx')"
                )
            )
        }
    assert names == {"daily_variance_model_date_idx", "clean_daily_bars_date_idx"}
