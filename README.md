# Volatility Risk Engine

Automated market risk analytics. The pipeline ingests daily OHLCV data for a basket of US
equities and indices, stores and transforms it in PostgreSQL, forecasts next-day volatility with
an ablation ladder of models (EWMA → GARCH(1,1) → HAR-RV → LightGBM), converts forecasts into
1-day Value-at-Risk with coverage backtesting, and feeds a Power BI dashboard.

> **Positioning.** This is a risk analytics project. The intellectual core is the distinction
> that volatility is predictable while returns are not — we forecast risk, never direction.
> Nothing in this repository is a trading signal, a price prediction, or investment advice.

## Status

Step 8 of 13 — the full ablation ladder: EWMA(0.94) → GARCH(1,1) → HAR-RV → LightGBM
(plus a LightGBM+VIX exogenous variant), one shared walk-forward harness (expanding
window, ≥3y train, monthly refits), evaluated with QLIKE (primary) and RMSE (secondary)
per ticker and on average. The roadmap lives in [CLAUDE.md](CLAUDE.md).

## Ablation results (v1)

Next-day variance forecasts, walk-forward only — no random splits, no tuning; every
forecast uses information through the previous session. `lgbm_vix` adds lagged ^VIX
level/change as exogenous regressors (reference data, per the ^VIX ruling).

**Modeling note (why the regressions fit log variance).** v1 fit the regression models
on variance *levels*; in calm regimes they emitted a handful of near-zero (HAR: even
negative, floored) forecasts, and QLIKE — asymmetric by design, punishing variance
under-forecasts hardest, which is the right asymmetry for risk work — blew up on
exactly those dates (JPM HAR-RV: 755.6 with them, 0.29 without). v2 therefore fits
`ln(variance)` and maps back with the lognormal half-variance correction
`exp(m + s^2/2)` (s^2 = training-residual variance in log space, re-estimated at each
refit); raw exponentiation would target the conditional *median* and systematically
under-forecast the mean — the direction QLIKE punishes most. HAR-RV uses Corsi's
log-log form (ln components as regressors, coefficients are elasticities): a log
target over *level* features put spike-day component values straight into the
exponent and produced astronomical over-forecasts — QLIKE's logarithmic over-forecast
penalty barely moved while RMSE detonated, the exact mirror image of the v1 pathology.
LightGBM keeps level features (trees split, they don't extrapolate). The 1e-8
positivity floor remains as a canary only: with log-space fits it should never bind
(expected floored count: 0).

**Proxy robustness.** Re-scored against the noisier squared-return proxy (kept in the
features layer for exactly this check), the ranking flips: GARCH leads (average QLIKE
1.54) and HAR-RV's sweep does not persist (1.67). The two proxies target different
variances — Garman-Klass measures the intraday range and excludes the overnight gap,
while close-to-close squared returns include it — so each model family wins on the
target it trains on. The forecast set feeding the VaR layer is chosen against the
VaR-relevant (close-to-close) target, not this table alone.

<!-- ABLATION:BEGIN -->
Evaluation set: the per-ticker INTERSECTION of every model's forecast dates — identical for all models by construction: n = 1756 sessions per ticker, 2019-07-15 to 2026-07-09. Realized-variance proxy: Garman-Klass. Lower is better; row-best in bold.

**QLIKE (primary)**

| ticker | ewma_094 | garch_11 | har_rv | lgbm | lgbm_vix |
|---|---|---|---|---|---|
| AAPL | 0.4201 | 0.3948 | **0.3006** | 0.4363 | 0.4033 |
| JPM | 0.3860 | 0.3304 | **0.2826** | 0.4436 | 0.3916 |
| MSFT | 0.4071 | 0.3805 | **0.2827** | 0.4067 | 0.3795 |
| NVDA | 0.4055 | 0.4148 | **0.2850** | 0.3758 | 0.3577 |
| TSLA | 0.3965 | 0.4518 | **0.2746** | 0.3429 | 0.3397 |
| XOM | 0.3371 | 0.3293 | **0.2477** | 0.3519 | 0.3325 |
| ^GSPC | 0.5767 | 0.5069 | **0.3791** | 0.5126 | 0.4504 |
| **AVERAGE** | 0.4184 | 0.4012 | **0.2932** | 0.4100 | 0.3793 |

**RMSE, annualized-vol percentage points (secondary)**

| ticker | ewma_094 | garch_11 | har_rv | lgbm | lgbm_vix |
|---|---|---|---|---|---|
| AAPL | 14.10 | 12.78 | **9.42** | 10.09 | 9.90 |
| JPM | 13.83 | 10.56 | **9.12** | 11.02 | 10.85 |
| MSFT | 13.05 | 11.62 | **8.38** | 9.50 | 9.49 |
| NVDA | 21.65 | 21.06 | **14.57** | 15.43 | 15.28 |
| TSLA | 27.09 | 28.58 | **18.51** | 19.38 | 18.96 |
| XOM | 12.95 | 12.24 | **9.66** | 11.00 | 10.70 |
| ^GSPC | 10.40 | 8.95 | **5.87** | 6.39 | 6.13 |
| **AVERAGE** | 16.15 | 15.11 | **10.79** | 11.83 | 11.62 |

*QLIKE(h, f) = h/f - ln(h/f) - 1 (Patton-class robust loss, normalized to 0 at f = h); dimensionless, lower is better. RMSE is in annualized-volatility percentage points, i.e. rmse(100·sqrt(252·h), 100·sqrt(252·f)). h = realized Garman-Klass variance, f = forecast; both are daily variances in return units.*
<!-- ABLATION:END -->

## Stack

Python 3.12 managed with [uv](https://docs.astral.sh/uv/) · PostgreSQL 16 · SQLAlchemy 2 +
psycopg 3 · pandas / numpy · pandera · arch · scikit-learn · LightGBM · pytest · ruff ·
GitHub Actions · Power BI

## Development setup

```bash
# prerequisite: uv (https://docs.astral.sh/uv/getting-started/installation/)
uv sync                    # create .venv and install locked dependencies
uv run pytest              # run the test suite
uv run ruff check .        # lint
uv run ruff format --check .  # formatting check
uv run pre-commit install  # enable git hooks
cp .env.example .env       # then fill in local credentials (never committed)
```

## Database setup (PostgreSQL 16)

Everything connects through `DATABASE_URL` in `.env`, so either route below works
unchanged. Pick one:

**Route A — native install (used on the primary dev machine).** Install PostgreSQL 16
via the [EDB Windows installer](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)
(or your OS package manager), create a role and databases, and point `DATABASE_URL` at it.
If another Postgres already owns port 5432, install on 5433 and reflect that in the URL:

```sql
CREATE ROLE volrisk LOGIN PASSWORD '...';
CREATE DATABASE volrisk OWNER volrisk;
CREATE DATABASE volrisk_test OWNER volrisk;   -- disposable, for integration tests
```

**Route B — Docker Compose.**

```bash
docker compose up -d       # postgres:16 with credentials from .env
docker compose ps          # wait until healthy
```

**Then, with either route:**

```bash
uv run python -m volrisk.ingest.backfill        # 10y OHLCV -> data/raw/*.parquet (validated)
uv run python -m volrisk.db.migrate             # apply db/migrations/*.sql (tracked, idempotent)
uv run python -m volrisk.db.load_raw            # upsert parquet -> raw.daily_bars (re-run: net 0)
uv run python -m volrisk.transform.cleaning     # calendar-align -> clean.daily_bars + gap report
uv run python -m volrisk.features.build         # window functions -> features.daily_features
uv run python -m volrisk.features.crosscheck    # SQL vs pandas recomputation, per ticker
uv run python -m volrisk.models.baselines       # walk-forward EWMA + GARCH -> forecasts schema
uv run --env-file .env pytest                   # includes DB integration tests
```

Every load stage upserts on the natural key `(ticker, trade_date)`, so re-running any
stage is idempotent; recent bars are revisable by design (a fetch during market hours
lands an in-progress bar, which later runs revise to final values and the cleaning
stage excludes until its session has closed).

**Repo invariant: the modeling layer never sees an in-progress bar.** A bar reaches
`clean` — and everything downstream of it — only after its exchange session has closed.

Calendar note: ^VIX is CBOE-listed; the XNYS calendar is used as a proxy for the whole
US basket. That is a deliberate simplification — its artifacts (e.g. a phantom ^VIX bar
on a market holiday) are surfaced and excluded by the cleaning stage's gap report.
