"""Compute the ablation table: QLIKE + RMSE per ticker x model, plus averages.

Usage::

    uv run python -m volrisk.evaluate.ablation [--write-readme]

Realized-variance proxy: features.gk_var at the forecast date (CLAUDE.md
primary proxy). For each ticker, models are evaluated on the COMMON dates
where every model has a forecast, so the numbers are directly comparable.
Per-(ticker, model) rows are upserted into forecasts.ablation_metrics; the
AVERAGE row shown in the tables is the unweighted mean over tickers, computed
at read time. ``--write-readme`` replaces the marked ablation section in
README.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine
from volrisk.db.loaders import upsert_ablation_metrics
from volrisk.evaluate.metrics import qlike, rmse_ann_vol_pct

logger = logging.getLogger(__name__)

#: Ladder display order; tables show whichever of these exist in the DB.
MODEL_ORDER = ("ewma_094", "garch_11", "har_rv", "lgbm", "lgbm_vix")

README_BEGIN = "<!-- ABLATION:BEGIN -->"
README_END = "<!-- ABLATION:END -->"


def load_forecasts_with_realized(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(
        text(
            "SELECT f.ticker, f.trade_date, f.model, f.var_forecast, x.gk_var AS realized"
            " FROM forecasts.daily_variance f"
            " JOIN features.daily_features x USING (ticker, trade_date)"
            " WHERE x.gk_var > 0"
            " ORDER BY f.ticker, f.trade_date"
        ),
        engine,
    )


def compute_ablation(engine: Engine) -> pd.DataFrame:
    """Per-(ticker, model) metrics on each ticker's common forecast dates."""
    df = load_forecasts_with_realized(engine)
    results = []
    for ticker, group in df.groupby("ticker", sort=True):
        wide = group.pivot(index="trade_date", columns="model", values="var_forecast").dropna()
        realized = (
            group.drop_duplicates("trade_date").set_index("trade_date")["realized"].loc[wide.index]
        )
        for model in wide.columns:
            results.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "n_obs": len(wide),
                    "qlike": qlike(realized.to_numpy(), wide[model].to_numpy()),
                    "rmse_ann_vol_pct": rmse_ann_vol_pct(
                        realized.to_numpy(), wide[model].to_numpy()
                    ),
                    "eval_start": wide.index.min(),
                    "eval_end": wide.index.max(),
                }
            )
    return pd.DataFrame(results)


def _metric_table(metrics: pd.DataFrame, value_col: str, fmt: str) -> str:
    models = [m for m in MODEL_ORDER if m in set(metrics["model"])]
    wide = metrics.pivot(index="ticker", columns="model", values=value_col)[models]
    wide.loc["**AVERAGE**"] = wide.mean()

    lines = ["| ticker | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for ticker, row in wide.iterrows():
        best = row.min()
        cells = [f"**{value:{fmt}}**" if value == best else f"{value:{fmt}}" for value in row]
        lines.append(f"| {ticker} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_report(metrics: pd.DataFrame) -> str:
    n_vals = metrics.groupby("ticker")["n_obs"].first()
    spans = metrics.groupby("ticker")[["eval_start", "eval_end"]].first()
    uniform = (
        n_vals.nunique() == 1
        and spans["eval_start"].nunique() == 1
        and spans["eval_end"].nunique() == 1
    )
    if uniform:
        meta = (
            f"Evaluation set: the per-ticker INTERSECTION of every model's forecast dates — "
            f"identical for all models by construction: n = {int(n_vals.iloc[0])} sessions "
            f"per ticker, {spans['eval_start'].iloc[0]} to {spans['eval_end'].iloc[0]}."
        )
    else:
        per = "; ".join(
            f"{t}: n={int(n_vals[t])}, {spans.loc[t, 'eval_start']}..{spans.loc[t, 'eval_end']}"
            for t in n_vals.index
        )
        meta = f"Evaluation set per ticker (intersection of every model's forecast dates): {per}."

    footnote = (
        "*QLIKE(h, f) = h/f - ln(h/f) - 1 (Patton-class robust loss, normalized to 0 at "
        "f = h); dimensionless, lower is better. RMSE is in annualized-volatility "
        "percentage points, i.e. rmse(100·sqrt(252·h), 100·sqrt(252·f)). h = realized "
        "Garman-Klass variance, f = forecast; both are daily variances in return units.*"
    )
    parts = [
        meta + " Realized-variance proxy: Garman-Klass. Lower is better; row-best in bold.",
        "",
        "**QLIKE (primary)**",
        "",
        _metric_table(metrics, "qlike", ".4f"),
        "",
        "**RMSE, annualized-vol percentage points (secondary)**",
        "",
        _metric_table(metrics, "rmse_ann_vol_pct", ".2f"),
        "",
        footnote,
    ]
    return "\n".join(parts)


def write_readme_section(markdown: str, readme_path: Path = Path("README.md")) -> None:
    content = readme_path.read_text(encoding="utf-8")
    if README_BEGIN not in content or README_END not in content:
        raise SystemExit(f"README markers {README_BEGIN} / {README_END} not found")
    head, rest = content.split(README_BEGIN, 1)
    _, tail = rest.split(README_END, 1)
    readme_path.write_text(
        f"{head}{README_BEGIN}\n{markdown}\n{README_END}{tail}", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and store the ablation table.")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    engine = get_engine()
    metrics = compute_ablation(engine)
    if metrics.empty:
        raise SystemExit("no forecasts found to evaluate")
    n = upsert_ablation_metrics(engine, metrics)
    logger.info("stored %d ablation rows", n)

    report = markdown_report(metrics)
    print()
    print(report)
    if args.write_readme:
        write_readme_section(report)
        print("\nREADME ablation section updated.")


if __name__ == "__main__":
    main()
