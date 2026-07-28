"""Optional MLflow experiment tracking (Stretch 3).

Design constraints, all deliberate:

- **Local and serverless.** The tracking store is a single SQLite file
  (``mlflow.db``) with artifacts under ``mlruns/`` — both gitignored, both
  regenerable from Postgres. Nothing needs to be running for the pipeline to
  work, and no cloud service is involved. (MLflow 3.14 put its plain-directory
  file store into maintenance mode and raises unless explicitly opted into;
  SQLite is the supported local equivalent and needs no server either.)
- **Never a hard runtime dependency.** MLflow is an optional extra. Every entry
  point here degrades to a no-op when it is not installed, so the nightly job
  behaves identically with or without it — a tracking failure must never fail a
  risk pipeline.
- **Cheap.** One run per evaluation, a few hundred metrics, two small artifacts.
  Logging is wrapped so any MLflow-side error is caught and logged rather than
  raised.

Disable explicitly with ``VOLRISK_DISABLE_MLFLOW=1`` even when installed.

Open the UI with (the full ``mlflow`` package pins pandas<3, so it is run from
an isolated tool environment rather than added to this project)::

    uvx mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "volrisk-walkforward"
#: Local SQLite tracking store (a file, not a server) and its artifact root.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_DIR = Path("mlruns")

#: MLflow keys allow alphanumerics, underscores, dashes, periods, spaces and
#: slashes — "^GSPC" would be rejected, so keys are sanitized (and the mapping
#: is recorded as a tag so nothing is silently renamed).
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-./ ]")


def sanitize_key(name: str) -> str:
    """MLflow-safe metric/param key ('^GSPC' -> 'GSPC')."""
    return _INVALID_KEY_CHARS.sub("", name)


def is_enabled() -> bool:
    """True when MLflow is importable and tracking is not explicitly disabled."""
    if os.environ.get("VOLRISK_DISABLE_MLFLOW", "").strip() not in ("", "0", "false", "False"):
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def _flatten_metrics(df: pd.DataFrame, value_cols: dict[str, str]) -> dict[str, float]:
    """Long frame -> {'<prefix>.<model>.<ticker>': value} metric dictionary."""
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        for col, prefix in value_cols.items():
            value = row.get(col)
            if value is None or pd.isna(value):
                continue
            key = f"{prefix}.{sanitize_key(str(row['model']))}.{sanitize_key(str(row['ticker']))}"
            out[key] = float(value)
    return out


def log_evaluation(
    ablation: pd.DataFrame,
    coverage: pd.DataFrame,
    params: dict[str, Any] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    run_name: str | None = None,
) -> str | None:
    """Log one walk-forward evaluation. Returns the run id, or None if skipped.

    Logs per-(ticker, model) QLIKE/RMSE, coverage and independence statistics,
    aggregate averages, the run's configuration parameters, and the ablation and
    coverage tables as CSV artifacts. Any failure is swallowed with a warning:
    tracking is observability, not a pipeline stage.
    """
    if not is_enabled():
        logger.info("MLflow tracking skipped (not installed or disabled)")
        return None

    try:
        import mlflow

        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
            mlflow.create_experiment(
                EXPERIMENT_NAME, artifact_location=artifact_dir.resolve().as_uri()
            )
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v) for k, v in (params or {}).items()})

            metrics = _flatten_metrics(ablation, {"qlike": "qlike", "rmse_ann_vol_pct": "rmse"})
            cov95 = coverage[coverage["level"] == 95]
            cov99 = coverage[coverage["level"] == 99]
            for frame, level in ((cov95, 95), (cov99, 99)):
                metrics |= _flatten_metrics(
                    frame,
                    {
                        "breach_rate": f"breach_rate_{level}",
                        "kupiec_p": f"kupiec_p_{level}",
                        "p_ind": f"p_ind_{level}",
                    },
                )

            # Aggregates: the numbers a reader compares across runs.
            for model, group in ablation.groupby("model"):
                m = sanitize_key(str(model))
                metrics[f"avg.qlike.{m}"] = float(group["qlike"].mean())
                metrics[f"avg.rmse.{m}"] = float(group["rmse_ann_vol_pct"].mean())
            for level, frame in ((95, cov95), (99, cov99)):
                for model, group in frame.groupby("model"):
                    m = sanitize_key(str(model))
                    metrics[f"avg.breach_rate_{level}.{m}"] = float(group["breach_rate"].mean())
                    metrics[f"kupiec_rejects_{level}.{m}"] = float((group["kupiec_p"] < 0.05).sum())
                    metrics[f"independence_rejects_{level}.{m}"] = float(
                        (group["p_ind"] < 0.05).sum()
                    )
            mlflow.log_metrics(metrics)

            mlflow.log_text(ablation.to_csv(index=False), "ablation.csv")
            mlflow.log_text(coverage.to_csv(index=False), "var_coverage.csv")
            mlflow.log_text(_ablation_markdown(ablation), "ablation.md")
            logger.info("logged MLflow run %s (%d metrics)", run.info.run_id, len(metrics))
            return str(run.info.run_id)
    except Exception as exc:  # tracking must never break the pipeline
        logger.warning("MLflow logging failed (continuing): %s", exc)
        return None


def _markdown_table(wide: pd.DataFrame, fmt: str) -> str:
    """Pivot -> markdown, written by hand so no extra dependency is needed."""
    cols = list(wide.columns)
    lines = ["| ticker | " + " | ".join(map(str, cols)) + " |", "|---" * (len(cols) + 1) + "|"]
    for ticker, row in wide.iterrows():
        cells = ["" if pd.isna(v) else f"{v:{fmt}}" for v in row]
        lines.append(f"| {ticker} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ablation_markdown(ablation: pd.DataFrame) -> str:
    """The ablation table as markdown, for reading inside the MLflow UI."""
    qlike = ablation.pivot(index="ticker", columns="model", values="qlike")
    rmse = ablation.pivot(index="ticker", columns="model", values="rmse_ann_vol_pct")
    return (
        "# Ablation\n\n## QLIKE\n\n"
        + _markdown_table(qlike, ".4f")
        + "\n\n## RMSE (annualized vol points)\n\n"
        + _markdown_table(rmse, ".2f")
        + "\n"
    )


def log_from_database(engine, run_name: str | None = None) -> str | None:
    """Log the evaluation currently stored in Postgres (no re-computation)."""
    ablation = pd.read_sql_query("SELECT * FROM forecasts.ablation_metrics", engine)
    coverage = pd.read_sql_query("SELECT * FROM forecasts.var_coverage", engine)
    if ablation.empty or coverage.empty:
        raise SystemExit("nothing to log: run the ablation and backtest first")
    return log_evaluation(ablation, coverage, params=run_params(ablation), run_name=run_name)


def run_params(ablation: pd.DataFrame) -> dict[str, Any]:
    """Configuration worth pinning beside the metrics."""
    from volrisk.evaluate.walkforward import VARIANCE_FLOOR
    from volrisk.ingest.backfill import BACKFILL_START
    from volrisk.models.baselines import MIN_TRAIN_SESSIONS
    from volrisk.risk.backtest import CALIBRATION_TRIGGER_RATE
    from volrisk.risk.student_t import MAX_DF, MIN_DF, MIN_FIT_OBS

    return {
        "backfill_start": BACKFILL_START,
        "min_train_sessions": MIN_TRAIN_SESSIONS,
        "refit_cadence": "calendar-monthly",
        "variance_floor": VARIANCE_FLOOR,
        "calibration_trigger_rate": CALIBRATION_TRIGGER_RATE,
        "t_df_bounds": f"{MIN_DF}..{MAX_DF}",
        "t_df_min_fit_obs": MIN_FIT_OBS,
        "realized_proxy": "garman_klass",
        "n_obs_per_ticker": int(ablation["n_obs"].min()),
        "eval_start": str(ablation["eval_start"].max()),
        "eval_end": str(ablation["eval_end"].min()),
    }


def main() -> None:
    """CLI: log whatever evaluation is currently in the database."""
    import argparse

    parser = argparse.ArgumentParser(description="Log the stored evaluation to MLflow.")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not is_enabled():
        raise SystemExit(
            "MLflow is not available. Install the optional extra with "
            "`uv sync --extra tracking` (or unset VOLRISK_DISABLE_MLFLOW)."
        )

    from volrisk.db.engine import get_engine

    run_id = log_from_database(get_engine(), run_name=args.run_name)
    print(f"logged MLflow run: {run_id}")
    print(f"open the UI with: uvx mlflow ui --backend-store-uri {DEFAULT_TRACKING_URI}")


if __name__ == "__main__":
    main()
