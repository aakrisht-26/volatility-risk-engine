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

Known v1 caveat: the regression models occasionally emit near-zero variance forecasts
(4–45 of ~1,756 dates per series), which QLIKE punishes roughly linearly in
realized/forecast — those few dates dominate their QLIKE means below. Excluding
forecasts under 1e-6, HAR-RV's QLIKE is 0.29–0.39, competitive with or better than
GARCH; the RMSE table is unaffected. v2 (pending review) will address forecast
positivity structurally (log-variance targets).

<!-- ABLATION:BEGIN -->
Walk-forward evaluation on each ticker's common forecast dates (~1756 sessions per ticker, 2019-07-15 to 2026-07-09). Realized-variance proxy: Garman-Klass. Lower is better; row-best in bold.

**QLIKE (primary)**

| ticker | ewma_094 | garch_11 | har_rv | lgbm | lgbm_vix |
|---|---|---|---|---|---|
| AAPL | 0.4201 | 0.3948 | **0.3000** | 184.3099 | 170.6866 |
| JPM | 0.3860 | **0.3304** | 755.9906 | 727.9801 | 677.6706 |
| MSFT | 0.4071 | 0.3805 | **0.2981** | 225.2434 | 49.8936 |
| NVDA | 0.4055 | 0.4148 | **0.2866** | 146.0983 | 199.0583 |
| TSLA | 0.3965 | 0.4518 | **0.2729** | 491.0889 | 350.4063 |
| XOM | 0.3371 | **0.3293** | 225.2810 | 296.4946 | 579.6450 |
| ^GSPC | 0.5767 | **0.5069** | 30.8317 | 199.4603 | 103.8150 |
| **AVERAGE** | 0.4184 | **0.4012** | 144.7516 | 324.3822 | 304.4536 |

**RMSE, annualized-vol percentage points (secondary)**

| ticker | ewma_094 | garch_11 | har_rv | lgbm | lgbm_vix |
|---|---|---|---|---|---|
| AAPL | 14.10 | 12.78 | **9.36** | 10.70 | 10.74 |
| JPM | 13.83 | 10.56 | **9.83** | 12.37 | 12.21 |
| MSFT | 13.05 | 11.62 | **8.57** | 10.40 | 10.45 |
| NVDA | 21.65 | 21.06 | **14.70** | 16.75 | 16.92 |
| TSLA | 27.09 | 28.58 | **18.64** | 21.58 | 20.65 |
| XOM | 12.95 | 12.24 | **10.11** | 12.26 | 11.94 |
| ^GSPC | 10.40 | 8.95 | **5.91** | 6.84 | 6.60 |
| **AVERAGE** | 16.15 | 15.11 | **11.02** | 12.98 | 12.79 |
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
