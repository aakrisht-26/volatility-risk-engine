"""VaR coverage backtest: parametric VaR + Kupiec for every model, all levels.

Usage::

    uv run python -m volrisk.risk.backtest [--write-readme]

Evaluates on the same per-ticker intersection window as the ablation (the dates
where every base model has a forecast), records n and span in the stored
metadata, and writes per-(ticker, model, level) coverage plus the sparse breach
series to the ``forecasts`` schema.

Conditional calibration (requirement 4): if pre-registered prediction (i) is
confirmed — the three GK-target models' average 95% breach rate >= 6.0% — the
runner also builds ``_cal`` variants that rescale each session-range variance by
a walk-forward, training-only ratio c = mean(r^2)/mean(gk_var) (expanding
window, re-estimated at each monthly refit boundary, strictly no look-ahead),
converting session-range variance to close-to-close variance before VaR.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine
from volrisk.db.loaders import store_var_results
from volrisk.risk.kupiec import kupiec_pof
from volrisk.risk.var import LEVELS, TAIL_PROB, breach_mask, var_threshold

logger = logging.getLogger(__name__)

BASE_MODEL_ORDER = ("ewma_094", "garch_11", "har_rv", "lgbm", "lgbm_vix")
GK_TARGET_MODELS = ("har_rv", "lgbm", "lgbm_vix")
CAL_SUFFIX = "_cal"

#: Prediction (i) confirmation criterion (pre-registered): the three GK-target
#: models' average observed 95% breach rate at or above 6.0% (>= 20% relative
#: excess over the 5% nominal).
CALIBRATION_TRIGGER_RATE = 0.06

README_BEGIN = "<!-- VAR:BEGIN -->"
README_END = "<!-- VAR:END -->"


def load_forecasts_and_returns(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(
        text(
            "SELECT f.ticker, f.trade_date, f.model, f.var_forecast, c.log_return"
            " FROM forecasts.daily_variance f"
            " JOIN clean.daily_bars c USING (ticker, trade_date)"
            " WHERE c.log_return IS NOT NULL"
            " ORDER BY f.ticker, f.trade_date"
        ),
        engine,
    )


def load_calibration_inputs(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(
        text(
            "SELECT ticker, trade_date, r2, gk_var FROM features.daily_features"
            " WHERE r2 IS NOT NULL AND gk_var > 0 ORDER BY ticker, trade_date"
        ),
        engine,
    )


def evaluate_coverage(
    returns: pd.Series, variance: pd.Series, level: int
) -> tuple[dict, pd.DataFrame]:
    """Coverage summary + breach-event rows for one (returns, variance) series."""
    thr = var_threshold(variance, level)
    r = returns.to_numpy(dtype=float)
    mask = breach_mask(r, thr)
    x, n = int(mask.sum()), len(r)
    k = kupiec_pof(n, x, TAIL_PROB[level])
    summary = {
        "level": level,
        "n_obs": n,
        "expected_breaches": TAIL_PROB[level] * n,
        "observed_breaches": x,
        "breach_rate": x / n,
        "kupiec_lr": k.lr_stat,
        "kupiec_p": k.p_value,
        "eval_start": returns.index[0],
        "eval_end": returns.index[-1],
    }
    events = pd.DataFrame(
        {
            "trade_date": returns.index[mask],
            "level": level,
            "log_return": r[mask],
            "var_threshold": thr[mask],
        }
    )
    return summary, events


def calibration_factors(feat: pd.DataFrame, forecast_dates: list[date]) -> pd.Series:
    """Walk-forward c = mean(r^2)/mean(gk_var) per forecast date, no look-ahead.

    Expanding training window, re-estimated only at each calendar-month boundary
    (matching the model refit cadence) and held constant within the month. The
    factor for month M uses feature rows strictly before M's first forecast
    session, so it never sees the dates it is applied to.
    """
    feat = feat.sort_values("trade_date")
    train_dates = feat["trade_date"].to_numpy()
    r2 = feat["r2"].to_numpy(dtype=float)
    gk = feat["gk_var"].to_numpy(dtype=float)

    factors: dict[date, float] = {}
    month: tuple[int, int] | None = None
    c = np.nan
    for d in sorted(forecast_dates):
        m = (d.year, d.month)
        if m != month:
            past = train_dates < np.datetime64(d)
            c = float(r2[past].mean() / gk[past].mean())
            month = m
        factors[d] = c
    return pd.Series(factors)


def prediction_i_confirmed(coverage: pd.DataFrame) -> bool:
    """GK-target models' average 95% breach rate >= the pre-registered trigger."""
    sub = coverage[(coverage["level"] == 95) & (coverage["model"].isin(GK_TARGET_MODELS))]
    return bool(sub["breach_rate"].mean() >= CALIBRATION_TRIGGER_RATE)


def _per_ticker_wide(fdf: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    per_ticker: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for ticker, g in fdf.groupby("ticker", sort=True):
        wide = g.pivot(index="trade_date", columns="model", values="var_forecast").dropna()
        ret = g.drop_duplicates("trade_date").set_index("trade_date")["log_return"].loc[wide.index]
        per_ticker[ticker] = (wide, ret)
    return per_ticker


def _collect(ticker: str, model: str, returns: pd.Series, variance: pd.Series):
    cov_rows, breach_frames = [], []
    for level in LEVELS:
        summary, events = evaluate_coverage(returns, variance, level)
        cov_rows.append({"ticker": ticker, "model": model, **summary})
        if not events.empty:
            breach_frames.append(events.assign(ticker=ticker, model=model))
    return cov_rows, breach_frames


def compute_backtest(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Compute coverage + breaches for all base models, then calibrated variants
    if prediction (i) is confirmed. Returns (coverage, breaches, calibrated)."""
    per_ticker = _per_ticker_wide(load_forecasts_and_returns(engine))

    cov_rows: list[dict] = []
    breach_frames: list[pd.DataFrame] = []
    for ticker, (wide, ret) in per_ticker.items():
        for model in wide.columns:
            rows, frames = _collect(ticker, model, ret, wide[model])
            cov_rows += rows
            breach_frames += frames
    coverage = pd.DataFrame(cov_rows)

    calibrated = prediction_i_confirmed(coverage)
    logger.info(
        "prediction (i) %s (GK-target avg 95%% breach rate = %.4f, trigger %.4f)",
        "CONFIRMED" if calibrated else "NOT confirmed",
        coverage[(coverage.level == 95) & (coverage.model.isin(GK_TARGET_MODELS))][
            "breach_rate"
        ].mean(),
        CALIBRATION_TRIGGER_RATE,
    )

    if calibrated:
        feats = load_calibration_inputs(engine)
        for ticker, (wide, ret) in per_ticker.items():
            tfeat = feats[feats["ticker"] == ticker]
            c = calibration_factors(tfeat, list(wide.index)).loc[wide.index]
            for model in GK_TARGET_MODELS:
                if model not in wide.columns:
                    continue
                rows, frames = _collect(ticker, model + CAL_SUFFIX, ret, wide[model] * c)
                cov_rows += rows
                breach_frames += frames
        coverage = pd.DataFrame(cov_rows)

    breaches = pd.concat(breach_frames, ignore_index=True) if breach_frames else pd.DataFrame()
    return coverage, breaches, calibrated


# --- reporting -------------------------------------------------------------


def _coverage_table(coverage: pd.DataFrame, level: int, models: list[str]) -> str:
    sub = coverage[coverage["level"] == level]
    present = [m for m in models if m in set(sub["model"])]
    obs = sub.pivot(index="ticker", columns="model", values="observed_breaches")[present]
    rate = sub.pivot(index="ticker", columns="model", values="breach_rate")[present]
    pval = sub.pivot(index="ticker", columns="model", values="kupiec_p")[present]

    lines = ["| ticker | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|"]
    for ticker in obs.index:
        cells = []
        for m in present:
            mark = " †" if pval.loc[ticker, m] < 0.05 else ""
            cells.append(f"{int(obs.loc[ticker, m])} ({rate.loc[ticker, m] * 100:.1f}%){mark}")
        lines.append(f"| {ticker} | " + " | ".join(cells) + " |")
    avg = obs.mean()
    lines.append("| **AVERAGE** | " + " | ".join(f"{avg[m]:.1f}" for m in present) + " |")
    rejections = {m: int((pval[m] < 0.05).sum()) for m in present}
    rej = " | ".join(f"{rejections[m]}" for m in present)
    lines.append(f"| **Kupiec rejects (/{len(obs)})** | " + rej + " |")
    return "\n".join(lines)


def render_report(coverage: pd.DataFrame, calibrated: bool) -> str:
    n = int(coverage["n_obs"].min())
    start, end = coverage["eval_start"].max(), coverage["eval_end"].min()
    parts = [
        f"Backtest window: per-ticker intersection of every base model's forecast dates, "
        f"n = {n} sessions, {start} to {end}. Cells show observed breaches (rate); "
        f"† = Kupiec rejects correct coverage at 5%.",
        "",
        f"**95% VaR** — expected {TAIL_PROB[95] * n:.1f} breaches / {n} sessions",
        "",
        _coverage_table(coverage, 95, list(BASE_MODEL_ORDER)),
        "",
        f"**99% VaR** — expected {TAIL_PROB[99] * n:.1f} breaches / {n} sessions",
        "",
        _coverage_table(coverage, 99, list(BASE_MODEL_ORDER)),
    ]
    if calibrated:
        cal_models = [m + CAL_SUFFIX for m in GK_TARGET_MODELS]
        parts += [
            "",
            "**Calibrated GK-target variants** (session-range variance rescaled to "
            "close-to-close by the walk-forward, training-only ratio c = mean(r^2)/mean(gk_var)).",
            "",
            f"*95% VaR — expected {TAIL_PROB[95] * n:.1f} breaches*",
            "",
            _coverage_table(coverage, 95, cal_models),
            "",
            f"*99% VaR — expected {TAIL_PROB[99] * n:.1f} breaches*",
            "",
            _coverage_table(coverage, 99, cal_models),
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
    parser = argparse.ArgumentParser(description="VaR coverage backtest (parametric + Kupiec).")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    engine = get_engine()
    coverage, breaches, calibrated = compute_backtest(engine)
    if coverage.empty:
        raise SystemExit("no forecasts found to backtest")
    n_cov, n_br = store_var_results(engine, coverage, breaches)
    logger.info("stored %d coverage rows, %d breach events", n_cov, n_br)

    report = render_report(coverage, calibrated)
    print()
    print(report)
    if args.write_readme:
        write_readme_section(report)
        print("\nREADME VaR section updated.")


if __name__ == "__main__":
    main()
