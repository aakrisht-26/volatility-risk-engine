# Volatility Risk Engine

Automated market risk analytics. The pipeline ingests daily OHLCV data for a basket of US
equities and indices, stores and transforms it in PostgreSQL, forecasts next-day volatility with
an ablation ladder of models (EWMA → GARCH(1,1) → HAR-RV → LightGBM), converts forecasts into
1-day Value-at-Risk with coverage backtesting, and feeds a Power BI dashboard.

> **Positioning.** This is a risk analytics project. The intellectual core is the distinction
> that volatility is predictable while returns are not — we forecast risk, never direction.
> Nothing in this repository is a trading signal, a price prediction, or investment advice.

## Status

Step 4 of 13 — Postgres raw layer (ingestion, validation, and idempotent DB load are in
place). The roadmap lives in [CLAUDE.md](CLAUDE.md).

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
uv run python -m volrisk.db.load_raw            # upsert parquet -> raw.daily_bars
uv run python -m volrisk.db.load_raw            # re-run: prints net new rows = 0
uv run --env-file .env pytest                   # includes DB integration tests
```
