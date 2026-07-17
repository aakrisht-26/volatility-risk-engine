# CLAUDE.md — Volatility Forecasting & Risk Analytics Engine

## What this project is
An automated market risk analytics system. It ingests daily OHLCV data for a basket of equities and indices via API, stores and transforms it in PostgreSQL, forecasts next-day volatility with an ablation ladder of models (EWMA → GARCH(1,1) → HAR-RV → LightGBM), converts forecasts into 1-day Value-at-Risk with coverage backtesting, and feeds a live Power BI dashboard. Flagship portfolio project — every choice should reflect industry-standard practice, not notebook shortcuts.

## Positioning rule (non-negotiable)
This is a RISK ANALYTICS project. Nothing in this repo — code, comments, docstrings, commit messages, README — may be framed as trading signals, price prediction, or investment advice. The intellectual core is the distinction: volatility is predictable, returns are not. We forecast risk, never direction.

## Working agreements (how to behave in this repo)
- Work ONE roadmap step at a time. Never start the next step without explicit approval from Aakrisht.
- After completing a step: summarize what was built, list files touched, give the exact commands to verify it works, then STOP and wait for review.
- When a decision has meaningful tradeoffs, present 2–3 options with a recommendation and ask. Do not silently choose.
- Briefly explain the "why" behind non-obvious engineering choices as you go — this build is also a learning exercise in industry practice.
- Small conventional commits (feat:, fix:, test:, docs:, chore:) per logical change; commit at minimum at the end of every step. The commit history is part of the portfolio — recruiters may read it, so messages stay professional and specific.
- Never commit secrets. `.env` is gitignored; keep `.env.example` current at all times.
- Tests must never hit the network. Use small fixture CSVs under `tests/fixtures/`.
- Re-running any pipeline stage must be idempotent — zero duplicate rows, ever. Loaders upsert on natural keys.
- Type hints on all public functions. Logging via Python's `logging` module with a configured formatter; no `print()` in library code (CLI entrypoints may print summaries).

## Locked tech decisions
- Python 3.11+, managed with uv (`uv venv`, `uv add`, `uv run`). Commit `uv.lock` and `.python-version` — builds must be reproducible.
- PostgreSQL 16 — Docker Compose preferred; native Windows installer is an acceptable fallback if Docker is a problem on this machine (ask before choosing).
- SQLAlchemy 2.x with psycopg v3 (`psycopg[binary]`, dialect `postgresql+psycopg://`) — current-generation driver.
- Plain versioned SQL files in `db/migrations/` (001_raw.sql, 002_clean.sql, …); alembic = stretch goal, not now.
- pandas / numpy for transforms; pandera for batch validation.
- arch (GARCH), scikit-learn (HAR-RV as linear regression), lightgbm.
- pandas-market-calendars for exchange calendars (XNYS now; NSE calendar if India is added).
- pytest for tests; ruff for lint + format; pre-commit wired to ruff; GitHub Actions for CI.
- Data providers: yfinance is PRIMARY, behind a provider interface (`providers/base.py`). Stooq CSV endpoint is the fallback provider. No yfinance calls anywhere outside `providers/`. This abstraction exists because yfinance is unofficial and occasionally breaks.
- MLflow: stretch goal at the final step only. Do not introduce earlier.

## Data decisions
- Phase-1 basket (US): ^GSPC, ^VIX, AAPL, MSFT, NVDA, JPM, XOM, TSLA
- Backfill: fixed inception **2016-07-11** (`BACKFILL_START`; the window does NOT slide — audit fix 2026-07-15, replacing the original "10 years back from today" which eroded landing-zone replayability). Trailing edge = today. Keep both adjusted and unadjusted close. Invariant: per-ticker parquet row count == raw.daily_bars row count after load.
- ^VIX ruling (2026-07-12, permanent — do not re-litigate): ^VIX is reference data only, excluded from the forecast/VaR universe. Rationale: spot VIX is not a holdable asset, so "risk of a VIX position" is ill-posed here. Its roles are implied-vol comparison and candidate exogenous feature (e.g. lagged level/change as regressors for other tickers' vol models).
- Daily bars carry `trade_date` as a DATE in exchange-local terms; no intraday timestamps. Any operational timestamps (job runs, load times) are stored UTC.
- Phase-2 basket (India, CONDITIONAL): ^NSEI, RELIANCE.NS, HDFCBANK.NS, INFY.NS, TCS.NS — added at Step 10 only if a data-quality audit passes (gap rate vs exchange calendar, zero-volume days, adjustment sanity around known splits). If the audit fails, document findings in the README and skip. Do not add India before Step 10.
- Same-day partial bars (recorded 2026-07-09, verified: pre-US-open vs post-open runs differ by one row per equity): a fetch during US market hours ingests today's in-progress bar. Raw parquet self-heals via wholesale replace. Requirements: (1) Step 4's loader must treat recent bars as revisable — the Step 11 nightly job re-fetches a trailing window (~5 trading days) and upserts, never only "yesterday"; (2) Step 5 defines the explicit policy for partial same-day bars; (3) Step 5's gap report reconciles same-day partials and exact per-ticker counts (incl. ^VIX's row surplus) against exchange calendars.

## Architecture / data flow
API → landing zone (immutable raw parquet in `data/raw/`, gitignored, replayable) → pandera validation → Postgres `raw` schema → cleaning → `clean` schema → feature engineering in SQL (window functions) → `features` schema → model training/forecasting in Python → forecasts + VaR written to `forecasts` schema → Power BI reads Postgres views directly.

DB schemas: `raw`, `clean`, `features`, `forecasts`. Natural key everywhere: (ticker, trade_date).

## Modeling definitions (keep consistent across the repo)
- Daily realized-variance proxy: Garman–Klass range estimator from OHLC (less noisy than squared returns); squared log returns kept as a robustness check.
- Target: next-day variance, annualized with 252 where displayed.
- HAR components: daily RV, weekly RV (5-day mean), monthly RV (22-day mean).
- EWMA baseline: RiskMetrics λ = 0.94.
- Evaluation: walk-forward only (expanding window, minimum ~3y train, refit monthly). Primary loss: QLIKE, exact form h/f − ln(h/f) − 1 (robust to noisy volatility proxies). Secondary: RMSE in annualized-vol points. Never random splits on time series.
- Feature-model targets (v2 ruling, 2026-07-12): HAR-RV and LightGBM fit **log variance** and map back with the lognormal half-variance retransformation exp(m + s²/2), s² = training-residual variance re-estimated at each refit. Rationale: level-space fits emitted near-zero forecasts that QLIKE (correctly) exploded on; raw exponentiation targets the conditional median and under-forecasts the mean — the direction risk work must not err in. HAR-RV is Corsi's **log-log** form (ln components as regressors; log-target-on-level-features provably explodes on vol spikes). LightGBM keeps level features (trees don't extrapolate). The 1e-8 positivity floor is a **canary**: expected floored count is 0; non-zero means investigate upstream.
- VaR: 1-day parametric 95% and 99% from each model's vol forecast (zero-mean normal; breach when r_d < −VaR); coverage tested with the Kupiec POF test. GK-target models get walk-forward calibrated variants (_cal: training-only mean(r²)/mean(gk_var) ratio) to convert session-range variance to close-to-close before VaR.

## Target repo structure
```
volatility-risk-engine/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docker-compose.yml          # postgres 16
├── .env.example                # DATABASE_URL etc.
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml    # ruff lint + format check + pytest on push/PR
├── db/migrations/              # 001_raw.sql, 002_clean.sql, ...
├── data/raw/                   # parquet landing zone (gitignored)
├── src/volrisk/
│   ├── config.py               # pydantic-settings, reads .env
│   ├── providers/              # base.py, yfinance_provider.py, stooq_provider.py
│   ├── ingest/                 # backfill.py, daily_update.py
│   ├── validate/               # schemas.py (pandera)
│   ├── db/                     # engine.py, loaders.py
│   ├── transform/              # cleaning.py, returns.py
│   ├── features/               # SQL feature build + runners
│   ├── models/                 # ewma.py, garch.py, har.py, lgbm.py
│   ├── evaluate/               # walkforward.py, metrics.py, ablation.py
│   └── risk/                   # var.py, kupiec.py
├── notebooks/                  # exploration only; nothing production lives here
└── tests/
    └── fixtures/
```

## Roadmap — execute ONE step per instruction, in order
Step 1 — Repo skeleton. uv project + pyproject, src layout, ruff config, pre-commit, pytest wired with one placeholder test, .gitignore, .env.example, README stub, git init + first commit, CI workflow (ruff check + ruff format --check + pytest). Acceptance: `uv run pytest`, `uv run ruff check .`, and `uv run ruff format --check .` pass locally; repo pushed to GitHub with green CI.

Step 2 — Providers + backfill. Provider ABC, YFinanceProvider, backfill script writing per-ticker raw parquet. Unit tests for parsing/normalization using fixtures (no network). Acceptance: 10y parquet for all 8 tickers with printed row counts; tests green.

Step 3 — Validation. Pandera schemas: dtypes, no null OHLC, positive prices, high ≥ low, high ≥ open/close ≥ low, unique (ticker, trade_date). Applied to every batch; failures raise with clear logs. Acceptance: full backfill passes; a deliberately corrupted fixture row fails in a test.

Step 4 — Postgres + raw load. docker-compose (or native install — ask), migration 001 for `raw`, idempotent upsert loader. Acceptance: raw row count == parquet row count; re-running the loader adds zero rows (prove with before/after counts).

Step 5 — Cleaning. Calendar alignment against each ticker's own exchange calendar (XNYS for the US basket), missing-day report, adjusted/unadjusted handling, log returns → `clean`. Acceptance: gap report printed; per-ticker check that summed log returns ≈ log(P_end/P_start).

Step 6 — SQL feature layer. Window functions: lagged returns, rolling realized vol (5/21/63d, annualized), Parkinson and Garman–Klass estimators, HAR components. Acceptance: `features` populated; Python recomputation of RV for one ticker matches SQL within tolerance (this cross-check IS the test).

Step 7 — Baselines. EWMA(0.94) and GARCH(1,1) per ticker via arch; walk-forward next-day variance forecasts written to `forecasts` with a model tag. Acceptance: full forecast series exist for both baselines.

Step 8 — Models + evaluation. HAR-RV regression and LightGBM on features; shared walk-forward harness; QLIKE + RMSE; ablation table (markdown + stored in DB). Acceptance: ablation table v1 in README covering all 4 models, per ticker + average.

Step 9 — Risk layer. 1-day 95/99% parametric VaR per model; Kupiec coverage backtest over the walk-forward span; breach series stored. Document the normal-distribution fat-tail limitation in the README (Student-t VaR = stretch). Acceptance: coverage table with expected vs observed breach rates and Kupiec p-values.

Step 10 — India audit (conditional expansion). Run the data-quality audit on NSE tickers; integrate + rerun pipeline if it passes, else document and skip.

Step 11 — Automation. Nightly incremental job: fetch latest bar → validate → load → incremental features → next-day forecasts + VaR → write back. ARCHITECTURE DECISION to make WITH Aakrisht here: (a) GitHub Actions cron + cloud Postgres (existing EC2 or RDS free tier; DATABASE_URL as an Actions secret) = fully live, strongest story; (b) Windows Task Scheduler + local Postgres = zero cost, less impressive. All connections already go through DATABASE_URL so either works.

Step 12 — Power BI. Produce SQL views for the dashboard plus a page-by-page spec with exact DAX measures (forecast vs realized, VaR breach tracker, model ablation, vol regime timeline). Aakrisht builds it in Power BI Desktop by hand. Also decided here: the VaR crown (garch_11 vs har_rv_cal, judged on the breach page) and any dashboard-pattern indexes (deferred from the 2026-07-15 audit — decided alongside the actual view queries, not before). Power BI points at **Neon** (the system of record); local Postgres 5433 remains the dev DB.

Step 13 — Polish. README restructured to lead with the ablation + VaR coverage results, architecture diagram, setup guide, limitations section, resume bullets. MLflow tracking as stretch.

## Open decisions (do not resolve unilaterally)
- (none currently)

## Resolved decisions (recorded; do not re-litigate)
- Step 11 (resolved 2026-07-17): architecture option (a) — GitHub Actions cron + **Neon serverless Postgres free tier** (not EC2, not RDS; Supabase is the fallback host if Neon surprises). Nightly job = full anchored re-fetch through a yfinance→Stooq fallback chain, full re-run of clean/features/forecasts/VaR (revisit incremental only if job time exceeds ~20 min), canaries promoted to exit codes. Stooq rows are adjusted-only: close == adj_close, flagged via raw.daily_bars.source.
- Landing-zone semantics under ephemeral runners (ruled 2026-07-17): the cloud DB is the **system of record**; a runner's parquet is deterministic staging reconstructable from the anchor; Aakrisht's local data/raw is the durable replay copy. The monotonic guard refuses shrinking parquet overwrites (--force only after investigation); trailing-window fallbacks land as dated increment files under data/raw/increments/, never overwriting the anchored zone.
- Step 4 (resolved 2026-07-10): native PostgreSQL 16.14 at E:\postgresql16, port **5433**, service postgresql-x64-16. A pre-existing PostgreSQL 18 at E:\postgresql owns 5432 — present and untouched, keep it that way. docker-compose.yml remains the documented alternative route.
- VaR crown (ruled 2026-07-13): no model crowned yet. The two live candidates are **garch_11** (native close-to-close, well-calibrated at 95%, boring-and-defensible) and **har_rv_cal** (best accuracy plus competitive coverage after honest walk-forward calibration). The decision is made at Step 12's VaR breach page, judged on how each reads there.
- Step 10 / India (ruled 2026-07-13): audit = GO on data quality (all 5 tickers; findings in README, audited 2026-07-13). Integration DEFERRED pending calendar handling; when integrated, the preferred path is option (a): augment the XNSE calendar with NSE's special sessions (Diwali Muhurat, Budget Saturdays) and ad-hoc closures, rather than treating bar dates as authoritative.
