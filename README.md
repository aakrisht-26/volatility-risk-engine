# Volatility Risk Engine

Automated market risk analytics. The pipeline ingests daily OHLCV data for a basket of US
equities and indices, stores and transforms it in PostgreSQL, forecasts next-day volatility with
an ablation ladder of models (EWMA → GARCH(1,1) → HAR-RV → LightGBM), converts forecasts into
1-day Value-at-Risk with coverage backtesting, and feeds a Power BI dashboard.

> **Positioning.** This is a risk analytics project. The intellectual core is the distinction
> that volatility is predictable while returns are not — we forecast risk, never direction.
> Nothing in this repository is a trading signal, a price prediction, or investment advice.

## Status

Step 6 of 13 — SQL feature layer: window functions over `clean` compute lagged returns,
rolling realized volatility (5/21/63d, annualized), Parkinson and Garman–Klass range
estimators, and HAR components, cross-checked column-by-column against an independent
pandas recomputation at machine epsilon. The roadmap lives in [CLAUDE.md](CLAUDE.md).

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
