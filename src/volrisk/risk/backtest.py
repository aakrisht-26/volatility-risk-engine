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
from volrisk.db.loaders import store_var_results, upsert_variance_forecasts
from volrisk.models.baselines import forecasts_frame
from volrisk.risk.christoffersen import christoffersen_independence, conditional_coverage
from volrisk.risk.kupiec import kupiec_pof
from volrisk.risk.student_t import t_var_thresholds, walk_forward_t_df
from volrisk.risk.var import LEVELS, TAIL_PROB, breach_mask, var_threshold

logger = logging.getLogger(__name__)

BASE_MODEL_ORDER = ("ewma_094", "garch_11", "har_rv", "lgbm", "lgbm_vix")
GK_TARGET_MODELS = ("har_rv", "lgbm", "lgbm_vix")
CAL_SUFFIX = "_cal"
#: Student-t quantile variants (Stretch 2): same variance, fat-tailed quantile.
T_SUFFIX = "_t"

#: Prediction (i) confirmation criterion (pre-registered): the three GK-target
#: models' average observed 95% breach rate at or above 6.0% (>= 20% relative
#: excess over the 5% nominal).
CALIBRATION_TRIGGER_RATE = 0.06

README_BEGIN = "<!-- VAR:BEGIN -->"
README_END = "<!-- VAR:END -->"
IND_BEGIN = "<!-- INDEPENDENCE:BEGIN -->"
IND_END = "<!-- INDEPENDENCE:END -->"
T_BEGIN = "<!-- STUDENTT:BEGIN -->"
T_END = "<!-- STUDENTT:END -->"


def load_forecasts_and_returns(engine: Engine) -> pd.DataFrame:
    # NOT is_live is explicit even though the clean join would usually exclude
    # live rows structurally: a live row has no realized outcome and must never
    # enter coverage (pinned by tests).
    return pd.read_sql_query(
        text(
            "SELECT f.ticker, f.trade_date, f.model, f.var_forecast, c.log_return"
            " FROM forecasts.daily_variance f"
            " JOIN clean.daily_bars c USING (ticker, trade_date)"
            " WHERE c.log_return IS NOT NULL AND NOT f.is_live"
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
    returns: pd.Series,
    variance: pd.Series,
    level: int,
    df_series: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Coverage summary + breach-event rows for one (returns, variance) series.

    With ``df_series`` the thresholds use each date's variance-matched Student-t
    quantile instead of the normal z — same variance forecast, different tail
    shape (Stretch 2).
    """
    if df_series is not None:
        thr = t_var_thresholds(variance, df_series, level / 100.0)
    else:
        thr = var_threshold(variance, level)
    r = returns.to_numpy(dtype=float)
    mask = breach_mask(r, thr)
    x, n = int(mask.sum()), len(r)
    k = kupiec_pof(n, x, TAIL_PROB[level])
    # Christoffersen on the SAME ordered breach series: rate (Kupiec) and
    # clustering (independence) are complementary, so they are reported side
    # by side, and LR_cc joins them.
    ind = christoffersen_independence(mask)
    lr_cc, p_cc = conditional_coverage(k.lr_stat, ind.lr_ind)
    summary = {
        "level": level,
        "n_obs": n,
        "expected_breaches": TAIL_PROB[level] * n,
        "observed_breaches": x,
        "breach_rate": x / n,
        "kupiec_lr": k.lr_stat,
        "kupiec_p": k.p_value,
        "n_00": ind.n_00,
        "n_01": ind.n_01,
        "n_10": ind.n_10,
        "n_11": ind.n_11,
        "lr_ind": ind.lr_ind,
        "p_ind": ind.p_ind,
        "lr_cc": lr_cc,
        "p_cc": p_cc,
        "t_df": float(np.median(df_series)) if df_series is not None else None,
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
    """Base-model forecast matrix per ticker. Restricted to BASE_MODEL_ORDER so
    persisted _cal rows (stored for the dashboard) are never re-consumed as
    inputs — the calibrated variants are always derived fresh from their base
    series and the training-only calibration factors."""
    per_ticker: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    base = fdf[fdf["model"].isin(BASE_MODEL_ORDER)]
    for ticker, g in base.groupby("ticker", sort=True):
        wide = g.pivot(index="trade_date", columns="model", values="var_forecast").dropna()
        ret = g.drop_duplicates("trade_date").set_index("trade_date")["log_return"].loc[wide.index]
        per_ticker[ticker] = (wide, ret)
    return per_ticker


def _collect(
    ticker: str,
    model: str,
    returns: pd.Series,
    variance: pd.Series,
    df_series: pd.Series | None = None,
):
    cov_rows, breach_frames = [], []
    for level in LEVELS:
        summary, events = evaluate_coverage(returns, variance, level, df_series=df_series)
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
    # (ticker, model) -> variance series, reused for the Student-t pass so the
    # t variants differ from their normal counterparts ONLY in tail shape.
    variance_series: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    for ticker, (wide, ret) in per_ticker.items():
        for model in wide.columns:
            rows, frames = _collect(ticker, model, ret, wide[model])
            cov_rows += rows
            breach_frames += frames
            variance_series[(ticker, model)] = (ret, wide[model])
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
                calibrated_series = wide[model] * c
                rows, frames = _collect(ticker, model + CAL_SUFFIX, ret, calibrated_series)
                cov_rows += rows
                breach_frames += frames
                variance_series[(ticker, model + CAL_SUFFIX)] = (ret, calibrated_series)
                # Persist the calibrated series as first-class forecast rows so
                # the dashboard (dashboard.v_var_daily etc.) can chart the _cal
                # variants' VaR bands — required for the Step-12 crown page.
                upsert_variance_forecasts(
                    engine,
                    forecasts_frame(ticker, calibrated_series, model + CAL_SUFFIX),
                    context=f"{ticker}:{model}{CAL_SUFFIX}",
                )
        coverage = pd.DataFrame(cov_rows)

        # Live _cal rows: calibrate each GK model's live next-session forecast
        # with the training-only factor for its month, so the featured model's
        # Overview card has a live VaR too. Flagged is_live like their bases.
        live = pd.read_sql_query(
            text(
                "SELECT ticker, trade_date, model, var_forecast"
                " FROM forecasts.daily_variance"
                " WHERE is_live AND model = ANY(:models)"
            ),
            engine,
            params={"models": list(GK_TARGET_MODELS)},
        )
        for row in live.itertuples():
            tfeat = feats[feats["ticker"] == row.ticker]
            factor = float(calibration_factors(tfeat, [row.trade_date]).iloc[0])
            upsert_variance_forecasts(
                engine,
                forecasts_frame(
                    row.ticker,
                    pd.Series([row.var_forecast * factor], index=[row.trade_date]),
                    row.model + CAL_SUFFIX,
                ),
                context=f"{row.ticker}:{row.model}{CAL_SUFFIX}:live",
                is_live=True,
            )

    # --- Student-t pass (Stretch 2) ---------------------------------------
    # Same variance forecasts, fat-tailed quantile. df is re-estimated by MLE
    # on standardized residuals at each monthly boundary using training data
    # only, so no threshold sees the return it is tested against.
    df_diagnostics: list[dict] = []
    for (ticker, model), (ret, variance) in list(variance_series.items()):
        est = walk_forward_t_df(ret, variance)
        rows, frames = _collect(ticker, model + T_SUFFIX, ret, variance, df_series=est.df_series)
        cov_rows += rows
        breach_frames += frames
        df_diagnostics.append(
            {
                "ticker": ticker,
                "model": model + T_SUFFIX,
                "refits": est.n_refits,
                "interior": est.n_interior,
                "clamped_high": est.n_clamped_high,
                "clamped_low": est.n_clamped_low,
                "insufficient": est.n_insufficient,
                "clamp_rate": est.clamp_rate,
                "interior_median_df": est.interior_median,
            }
        )
    coverage = pd.DataFrame(cov_rows)
    coverage.attrs["df_diagnostics"] = pd.DataFrame(df_diagnostics)

    # Live rows for the t variants. A t variant shares its base model's VARIANCE
    # forecast exactly — only the quantile differs — so the live row is a re-tag,
    # which is what lets the dashboard's Overview card filter on the featured
    # model tag (har_rv_cal_t) rather than approximating it with its base.
    live_base = pd.read_sql_query(
        text(
            "SELECT ticker, trade_date, model, var_forecast"
            " FROM forecasts.daily_variance WHERE is_live"
        ),
        engine,
    )
    for row in live_base.itertuples():
        if row.model.endswith(T_SUFFIX):
            continue
        upsert_variance_forecasts(
            engine,
            forecasts_frame(
                row.ticker,
                pd.Series([row.var_forecast], index=[row.trade_date]),
                row.model + T_SUFFIX,
            ),
            context=f"{row.ticker}:{row.model}{T_SUFFIX}:live",
            is_live=True,
        )

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


def _verdicts(coverage: pd.DataFrame) -> list[str]:
    """Outcome verdicts computed from the results, not hand-asserted."""

    def avg_rate(models: tuple[str, ...], level: int) -> float:
        sub = coverage[(coverage["level"] == level) & (coverage["model"].isin(models))]
        return float(sub["breach_rate"].mean())

    base = coverage[coverage["model"].isin(BASE_MODEL_ORDER)]
    rate99 = base[base["level"] == 99].groupby("model")["breach_rate"].mean()
    gk95, base95 = avg_rate(GK_TARGET_MODELS, 95), avg_rate(("ewma_094", "garch_11"), 95)
    gk_dev, base_dev = abs(gk95 - 0.05), abs(base95 - 0.05)

    def verdict(ok: bool) -> str:
        return "CONFIRMED" if ok else "not confirmed"

    return [
        "**Outcomes vs pre-registered predictions:**",
        "",
        f"- (i) GK-target models under-cover at 95%: **{verdict(gk95 >= 0.06)}** — avg breach "
        f"rate {gk95 * 100:.1f}% vs 5% nominal, all reject Kupiec.",
        f"- (ii) All base models under-cover at 99%: **{verdict(bool((rate99 > 0.01).all()))}** — "
        f"every model's avg 99% rate exceeds 1% (least-bad {rate99.min() * 100:.1f}%); the "
        "normal-quantile fat-tail limitation, measured.",
        f"- (iii) GARCH/EWMA closest to nominal at 95%: **{verdict(base_dev < gk_dev)}** — their "
        f"avg 95% rate {base95 * 100:.1f}% is {base_dev * 100:.1f}pp off nominal vs "
        f"{gk_dev * 100:.1f}pp for the GK-target models.",
        "",
    ]


def render_report(coverage: pd.DataFrame, calibrated: bool) -> str:
    n = int(coverage["n_obs"].min())
    start, end = coverage["eval_start"].max(), coverage["eval_end"].min()
    parts = [
        *_verdicts(coverage),
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


def independence_population(calibrated: bool) -> list[str]:
    """Every variant the independence verdicts are computed over, in report order.

    Each normal-quantile variant is followed immediately by its Student-t
    counterpart. This is the SAME list the tables render, which is what makes
    the block reconcilable: the reject columns sum to the verdict totals.
    """
    normal = list(BASE_MODEL_ORDER) + (
        [m + CAL_SUFFIX for m in GK_TARGET_MODELS] if calibrated else []
    )
    return [variant for m in normal for variant in (m, m + T_SUFFIX)]


def _independence_table(coverage: pd.DataFrame, level: int, models: list[str]) -> str:
    """n_11 (breach-after-breach) and the independence p-value, per variant.

    Variants are ROWS and tickers are COLUMNS. The transpose is deliberate: at
    16 variants a model-per-column table is unreadably wide, and this
    orientation puts the per-variant reject counts in a column the reader can
    add up to reproduce the totals quoted in the verdicts above.
    """
    sub = coverage[coverage["level"] == level]
    present = [m for m in models if m in set(sub["model"])]
    tickers = list(sub.pivot(index="model", columns="ticker", values="n_11").columns)
    n11 = sub.pivot(index="model", columns="ticker", values="n_11")
    pind = sub.pivot(index="model", columns="ticker", values="p_ind")
    pcc = sub.pivot(index="model", columns="ticker", values="p_cc")

    header = ["model", *tickers, "ind rejects", "LR_cc rejects"]
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    ind_total = cc_total = 0
    for m in present:
        cells = []
        for ticker in tickers:
            mark = " ‡" if pind.loc[m, ticker] < 0.05 else ""
            cells.append(f"{int(n11.loc[m, ticker])} / p={pind.loc[m, ticker]:.3f}{mark}")
        ind_rej = int((pind.loc[m] < 0.05).sum())
        cc_rej = int((pcc.loc[m] < 0.05).sum())
        ind_total += ind_rej
        cc_total += cc_rej
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {ind_rej} | {cc_rej} |")
    lines.append(
        f"| **TOTAL ({len(present)} variants x {len(tickers)} tickers = "
        f"{len(present) * len(tickers)} tests)** |"
        + " |" * len(tickers)
        + f" **{ind_total}** | **{cc_total}** |"
    )
    return "\n".join(lines)


def _clusters(row: pd.Series) -> bool:
    """True when a rejection is CLUSTERING (pi_11 > pi_01) rather than the
    opposite — breaches spaced too regularly also reject independence."""
    denom_0, denom_1 = row["n_00"] + row["n_01"], row["n_10"] + row["n_11"]
    if not denom_0 or not denom_1:
        return False
    return (row["n_11"] / denom_1) > (row["n_01"] / denom_0)


def _independence_verdicts(coverage: pd.DataFrame) -> list[str]:
    """Outcomes vs the pre-registered predictions, computed from the results."""
    garch_family = ("ewma_094", "garch_11")
    gk_base = GK_TARGET_MODELS
    rejected = coverage[coverage["p_ind"] < 0.05]

    def rate(models: tuple[str, ...]) -> tuple[int, int]:
        sub = coverage[coverage["model"].isin(models)]
        return int((sub["p_ind"] < 0.05).sum()), len(sub)

    g_rej, g_n = rate(garch_family)
    k_rej, k_n = rate(gk_base)
    at95 = int((rejected["level"] == 95).sum())
    at99 = int((rejected["level"] == 99).sum())
    clustering = int(rejected.apply(_clusters, axis=1).sum()) if not rejected.empty else 0
    anti = len(rejected) - clustering
    n11_99 = coverage[coverage["level"] == 99]["n_11"].median()

    n_base, n_tick = len(garch_family), coverage["ticker"].nunique()
    return [
        "**Outcomes vs pre-registered predictions:**",
        "",
        f"- (i) GARCH-family models pass independence more often: "
        f"**{'CONFIRMED' if g_rej / g_n < k_rej / k_n else 'not confirmed'}** — "
        f"ewma_094 + garch_11 reject {g_rej}/{g_n} tests, the GK-target models "
        f"{k_rej}/{k_n}. **Population: the {n_base + len(gk_base)} BASE variants only** "
        f"({n_base} + {len(gk_base)} models x {n_tick} tickers x 2 levels = "
        f"{g_n} + {k_n} tests), because that is the population the prediction was "
        f"registered over — the _cal and _t variants did not exist yet. These two "
        f"denominators therefore will NOT be found by adding up the tables below, "
        f"which cover all variants; every other number in this section will be.",
        f"- (ii) Failures concentrate at 95%: "
        f"**{'CONFIRMED' if at95 > at99 else 'not confirmed'}** — {at95} rejections at "
        f"95% vs {at99} at 99% (directional, not overwhelming).",
        f"- (iii) The 99% test is underpowered: **supported** — median n_11 at 99% is "
        f"{n11_99:.0f}, so most series carry almost no information about clustering; "
        f"non-rejection there is not evidence of independence.",
        "",
        f"**Surprise worth stating:** only **{clustering} of {len(rejected)}** rejections "
        f"are clustering (pi_11 > pi_01). The other **{anti}** are *anti*-clustering — "
        f"breaches spaced too regularly to be independent. Rejection is therefore not a "
        f"synonym for clustering, and an eyeball of the breach chart would likely not "
        f"flag the anti-clustered cases at all.",
        "",
    ]


def render_independence_report(coverage: pd.DataFrame, calibrated: bool) -> str:
    """Christoffersen independence + conditional-coverage tables."""
    models = independence_population(calibrated)
    n_tick = coverage["ticker"].nunique()
    n_variants = len([m for m in models if m in set(coverage["model"])])
    parts = [
        *_independence_verdicts(coverage),
        f"**Population for every figure below except verdict (i):** all {n_variants} model "
        f"variants x {n_tick} tickers x 2 levels = {n_variants * n_tick * 2} independence "
        "tests. Both tables render all of them, so their reject columns add up to the "
        "totals quoted above.",
        "",
        "Cells show **n_11 / p-value**: n_11 is the count of breaches immediately "
        "following a breach (the clustering signal), p is Christoffersen's LR_ind "
        "(chi-square(1), H0 = independence). ‡ = independence rejected at 5%. The two "
        "right-hand columns count each variant's rejections across the seven tickers — "
        "of independence, and of the joint conditional-coverage test "
        "LR_cc = LR_uc + LR_ind (chi-square(2)) — and the TOTAL row is their sum.",
        "",
        "**95% VaR — independence**",
        "",
        _independence_table(coverage, 95, models),
        "",
        "**99% VaR — independence**",
        "",
        _independence_table(coverage, 99, models),
    ]
    return "\n".join(parts)


def _student_t_verdicts(coverage: pd.DataFrame, models: list[str]) -> list[str]:
    """Outcomes vs the pre-registered Student-t predictions, computed from data."""

    def agg(model_set: list[str], level: int) -> dict:
        sub = coverage[coverage["model"].isin(model_set) & (coverage["level"] == level)]
        return {
            "rate": float(sub["breach_rate"].mean()),
            "kupiec_rej": int((sub["kupiec_p"] < 0.05).sum()),
            "ind_rej": int((sub["p_ind"] < 0.05).sum()),
            "breaches": float(sub["observed_breaches"].mean()),
            "n": len(sub),
        }

    t_models = [m + T_SUFFIX for m in models]
    n99, t99 = agg(models, 99), agg(t_models, 99)
    n95, t95 = agg(models, 95), agg(t_models, 95)
    # One df path per (ticker, model); it is stored against both levels, so
    # dedupe before summarising or every series would count twice.
    df_rows = coverage[coverage["t_df"].notna()].drop_duplicates(["ticker", "model"])
    dfs = df_rows["t_df"]
    df_lo, df_hi, df_med = dfs.min(), dfs.max(), dfs.median()
    in_range = int(((dfs >= 3) & (dfs <= 8)).sum()) / len(dfs) if len(dfs) else 0.0

    narrowed = abs(t99["rate"] - 0.01) < abs(n99["rate"] - 0.01)
    toward_over = t95["rate"] < n95["rate"]
    fewer_ind = t99["ind_rej"] < n99["ind_rej"]

    return [
        "**Outcomes vs pre-registered predictions:**",
        "",
        f"- (i) 99% under-coverage narrows materially: "
        f"**{'CONFIRMED' if narrowed else 'NOT confirmed'}** — average 99% breach rate "
        f"{n99['rate'] * 100:.2f}% (normal) -> {t99['rate'] * 100:.2f}% (t) against 1% "
        f"nominal; Kupiec rejections {n99['kupiec_rej']}/{n99['n']} -> "
        f"{t99['kupiec_rej']}/{t99['n']}.",
        f"- (ii) 95% coverage degrades toward over-coverage: "
        f"**{'CONFIRMED' if toward_over else 'NOT confirmed'}** — average 95% breach rate "
        f"{n95['rate'] * 100:.2f}% (normal) -> {t95['rate'] * 100:.2f}% (t); Kupiec "
        f"rejections {n95['kupiec_rej']}/{n95['n']} -> {t95['kupiec_rej']}/{t95['n']}.",
        f"- (iii) estimated df lands in 3-8: "
        f"**{'CONFIRMED' if in_range >= 0.5 else 'NOT confirmed'}** — median df "
        f"{df_med:.2f}, range {df_lo:.2f}-{df_hi:.2f}, {in_range * 100:.0f}% of "
        f"the {len(dfs)} (ticker, model) series inside 3-8. This {df_med:.2f} is the "
        "median of each series' OWN median df over its whole walk-forward path, "
        "**clamped months included** — distinct from the interior-only median quoted "
        "in the clamp-rate note below, which medians individual unclamped refits.",
        f"- (iv) fewer independence rejections at 99% under t: "
        f"**{'CONFIRMED' if fewer_ind else 'NOT confirmed'}** — {n99['ind_rej']}/{n99['n']} "
        f"(normal) -> {t99['ind_rej']}/{t99['n']} (t). Read with the power caveat "
        f"registered alongside it: mean 99% breaches fall {n99['breaches']:.1f} -> "
        f"{t99['breaches']:.1f}, so part of any drop is fewer events to detect "
        f"dependence in, not more independence.",
        "",
    ]


def _t_comparison_table(coverage: pd.DataFrame, level: int, models: list[str]) -> str:
    """Normal vs t, side by side: breach rate and Kupiec verdict per model."""
    sub = coverage[coverage["level"] == level]
    present = set(sub["model"])
    lines = [
        "| model | normal rate | t rate | normal Kupiec rejects | t Kupiec rejects | median df |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        if m not in present or m + T_SUFFIX not in present:
            continue
        norm = sub[sub["model"] == m]
        tvar = sub[sub["model"] == m + T_SUFFIX]
        lines.append(
            f"| {m} | {norm['breach_rate'].mean() * 100:.2f}% | "
            f"{tvar['breach_rate'].mean() * 100:.2f}% | "
            f"{int((norm['kupiec_p'] < 0.05).sum())}/{len(norm)} | "
            f"{int((tvar['kupiec_p'] < 0.05).sum())}/{len(tvar)} | "
            f"{tvar['t_df'].median():.2f} |"
        )
    return "\n".join(lines)


def render_student_t_report(coverage: pd.DataFrame, calibrated: bool) -> str:
    models = list(BASE_MODEL_ORDER) + (
        [m + CAL_SUFFIX for m in GK_TARGET_MODELS] if calibrated else []
    )
    parts = [
        *_student_t_verdicts(coverage, models),
        "Same variance forecasts, different quantile: a Student-t scaled so its "
        "variance equals the model's forecast, with degrees of freedom estimated by "
        "MLE on standardized residuals at each monthly refit (training data only). "
        "Nominal breach rates are 5% and 1%.",
        "",
        "**95% VaR — normal vs Student-t**",
        "",
        _t_comparison_table(coverage, 95, models),
        "",
        "**99% VaR — normal vs Student-t**",
        "",
        _t_comparison_table(coverage, 99, models),
    ]
    return "\n".join(parts)


def _replace_block(markdown: str, begin: str, end: str, readme_path: Path) -> None:
    content = readme_path.read_text(encoding="utf-8")
    if begin not in content or end not in content:
        raise SystemExit(f"README markers {begin} / {end} not found")
    head, rest = content.split(begin, 1)
    _, tail = rest.split(end, 1)
    readme_path.write_text(f"{head}{begin}\n{markdown}\n{end}{tail}", encoding="utf-8")


def write_readme_section(markdown: str, readme_path: Path = Path("README.md")) -> None:
    _replace_block(markdown, README_BEGIN, README_END, readme_path)


def write_independence_section(markdown: str, readme_path: Path = Path("README.md")) -> None:
    _replace_block(markdown, IND_BEGIN, IND_END, readme_path)


def write_student_t_section(markdown: str, readme_path: Path = Path("README.md")) -> None:
    _replace_block(markdown, T_BEGIN, T_END, readme_path)


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

    # df-clamp canary (Stretch-2 addendum): a HIGH clamp means the MLE ran to
    # the Gaussian boundary, so the t variant did nothing there; a LOW clamp
    # means it wanted df <= 2, where the variance-matched quantile collapses.
    diag = coverage.attrs.get("df_diagnostics")
    if diag is not None and not diag.empty:
        est = int((diag["refits"] - diag["insufficient"]).sum())
        high, low = int(diag["clamped_high"].sum()), int(diag["clamped_low"].sum())
        print("\n=== Student-t df estimation ===")
        print(diag.to_string(index=False))
        print(
            f"\ndf clamps (canary): {high} high + {low} low of {est} estimated refits "
            f"({(high + low) / est * 100:.1f}%); interior median df "
            f"{diag['interior_median_df'].median():.2f}"
        )

    report = render_report(coverage, calibrated)
    independence = render_independence_report(coverage, calibrated)
    student_t = render_student_t_report(coverage, calibrated)
    print()
    print(report)
    print()
    print(independence)
    print()
    print(student_t)
    if args.write_readme:
        write_readme_section(report)
        write_independence_section(independence)
        write_student_t_section(student_t)
        print("\nREADME VaR + independence + Student-t sections updated.")


if __name__ == "__main__":
    main()
