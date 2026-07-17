# Volatility Risk Engine

[![CI](https://github.com/aakrisht-26/volatility-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aakrisht-26/volatility-risk-engine/actions/workflows/ci.yml)
[![Nightly pipeline](https://github.com/aakrisht-26/volatility-risk-engine/actions/workflows/nightly.yml/badge.svg)](https://github.com/aakrisht-26/volatility-risk-engine/actions/workflows/nightly.yml)

Automated market risk analytics. The pipeline ingests daily OHLCV data for a basket of US
equities and indices, stores and transforms it in PostgreSQL, forecasts next-day volatility with
an ablation ladder of models (EWMA → GARCH(1,1) → HAR-RV → LightGBM), converts forecasts into
1-day Value-at-Risk with coverage backtesting, and feeds a Power BI dashboard.

> **Positioning.** This is a risk analytics project. The intellectual core is the distinction
> that volatility is predictable while returns are not — we forecast risk, never direction.
> Nothing in this repository is a trading signal, a price prediction, or investment advice.

## Status

Step 10 of 13 — India NSE data-quality audit (conditional gate). The five Phase-2 NSE
tickers were audited for gap rate, zero-volume days, and adjustment sanity; the data
passed, and integration awaits review of the one calendar caveat below. The forecasting
pipeline through Step 9 is unchanged. The roadmap lives in [CLAUDE.md](CLAUDE.md).

## India NSE audit (Step 10)

A conditional data-quality gate on the Phase-2 basket (^NSEI, RELIANCE.NS, HDFCBANK.NS,
INFY.NS, TCS.NS) — **an audit only; nothing here is integrated into the pipeline.** Run
with `uv run python -m volrisk.audit.nse` (10y daily bars vs the `XNSE` calendar).
**Audited 2026-07-13** — bar counts below reflect that fetch date and drift with re-runs.

| ticker | bars | gap rate¹ | special sessions² | zero-volume | close/adj (today) | max raw move | split jumps³ | verdict |
|---|---|---|---|---|---|---|---|---|
| ^NSEI | 2,463 | 0.36% | 6 | 1.2% | 1.0000 | 13.9% | 0 | GO |
| RELIANCE.NS | 2,472 | 0.20% | 11 | 0.2% | 1.0000 | 14.1% | 0 | GO |
| HDFCBANK.NS | 2,472 | 0.20% | 11 | 0.2% | 1.0000 | 13.5% | 0 | GO |
| INFY.NS | 2,472 | 0.20% | 11 | 0.2% | 1.0000 | 17.7% | 0 | GO |
| TCS.NS | 2,472 | 0.20% | 11 | 0.2% | 1.0000 | 9.9% | 0 | GO |

¹ calendar sessions with no bar / expected. ² bars on days the calendar marks closed.
³ single-day raw moves >35% — a nonzero count would flag an *unadjusted* corporate action.

**Adjustment sanity — clean.** Every known corporate action is correctly back-adjusted:
RELIANCE 1:1 bonus (2017-09), TCS 1:1 (2018-06), INFY 1:1 (2018-09), HDFCBANK 2:1
face-value split (2019-09) all show a continuous adjusted series, `close/adj_close`
converges to exactly 1.0000 today, and there are **zero** split-sized raw jumps across
40 stock-years — so no corporate action was missed. (yfinance's "Close" is itself
split-adjusted; only dividends separate it from "Adj Close" — the same property the US
basket already has.)

**Zero-volume — negligible.** 5 days per equity (0.2%); ^NSEI's 1.2% is index volume,
which is not a tradability signal and would be reference-data-only anyway (same role as
^VIX).

**The one real caveat — the calendar, not the prices.** The gap-rate and special-session
counts are **not** data holes; they are `pandas-market-calendars` `XNSE` disagreeing with
NSE's actual trading days. The 11 "special sessions" per equity are real NSE sessions the
calendar omits — **Diwali Muhurat** trading (2021-11-04, 2022-10-24, 2025-10-21, …) and
special **Budget-day Saturday** sessions (2025-02-01); the 5 "missing" sessions are
**ad-hoc NSE closures** the calendar didn't know (2024-01-22 Ram Mandir inauguration,
2024-11-20 Maharashtra elections). The price data tracks real NSE days *more* accurately
than the calendar. Consequence for integration: Step 5's cleaning excludes non-session
rows, which would wrongly drop ~11 genuine sessions per NSE ticker — so NSE integration
is **not** a drop-in. It needs a calendar decision first: augment `XNSE` with the special
sessions and ad-hoc holidays, or treat the fetched bar dates as authoritative NSE session
membership.

**Recommendation: GO on data quality, pending your call on calendar handling.** The
prices, volumes, and adjustments are clean and modelable across all five tickers. The
only integration cost is the bounded (~0.6% of days) calendar-metadata gap above, with
two clear fixes. Integration is **not** performed until approved.

## Value-at-Risk coverage (Step 9)

## Value-at-Risk coverage (Step 9)

1-day parametric VaR at 95% and 99% for every forecast model, backtested with the
Kupiec POF test on the same per-ticker intersection window as the ablation
(n = 1,756, 2019-07-15 to 2026-07-09).

**Method.** VaR_α(d) = z_α · sqrt(var_forecast(d)), assuming a **zero-mean** normal
1-day return — over one trading day the drift (~1e-4) is negligible next to volatility
(~1e-2), the standard 1-day parametric-VaR assumption. z₉₅ = 1.645, z₉₉ = 2.326. VaR is
a positive loss magnitude in log-return units. **Breach convention:** session d breaches
when the realized close-to-close log return r_d < −VaR_α(d) — the long-position loss
tail. Under the model P(breach) = 1 − α exactly, so expected breaches are 87.8 at 95%
and 17.6 at 99%.

**Pre-registered predictions** (committed before any results were computed — see git
history; the results block below was empty in the registering commit):

1. **(i)** The GK-target models (`har_rv`, `lgbm`, `lgbm_vix`) **under-cover at both
   levels** — their σ is *session-range* (Garman–Klass) volatility, which omits the
   overnight gap, so it understates close-to-close risk and produces more breaches than
   nominal.
2. **(ii)** **All models under-cover at 99%** — a normal quantile cannot reach the fat
   tails of real returns, so the 99% VaR is too small for every model.
3. **(iii)** **GARCH and EWMA sit closest to nominal at 95%** — their σ is already
   close-to-close, so at the level where the normal tail is least wrong they should be
   near-calibrated.

**Confirmation criterion for (i)** (pre-registered): the three GK-target models'
*average observed 95% breach rate* ≥ 6.0% (a ≥20% relative excess over the 5% nominal).
If met, the runner builds calibrated variants (`har_rv_cal`, `lgbm_cal`, `lgbm_vix_cal`)
that multiply each session-range variance by a per-ticker ratio
c = mean(r²)/mean(gk_var), estimated on **training data only** at each monthly refit
(expanding window, no look-ahead), converting session-range variance to close-to-close
variance, and re-runs Kupiec on them.

<!-- VAR:BEGIN -->
**Outcomes vs pre-registered predictions:**

- (i) GK-target models under-cover at 95%: **CONFIRMED** — avg breach rate 9.8% vs 5% nominal, all reject Kupiec.
- (ii) All base models under-cover at 99%: **CONFIRMED** — every model's avg 99% rate exceeds 1% (least-bad 1.9%); the normal-quantile fat-tail limitation, measured.
- (iii) GARCH/EWMA closest to nominal at 95%: **CONFIRMED** — their avg 95% rate 5.3% is 0.3pp off nominal vs 4.8pp for the GK-target models.

Backtest window: per-ticker intersection of every base model's forecast dates, n = 1756 sessions, 2019-07-15 to 2026-07-09. Cells show observed breaches (rate); † = Kupiec rejects correct coverage at 5%.

**95% VaR** — expected 87.8 breaches / 1756 sessions

| ticker | ewma_094 | garch_11 | har_rv | lgbm | lgbm_vix |
|---|---|---|---|---|---|
| AAPL | 88 (5.0%) | 85 (4.8%) | 135 (7.7%) † | 183 (10.4%) † | 174 (9.9%) † |
| JPM | 100 (5.7%) | 97 (5.5%) | 129 (7.3%) † | 167 (9.5%) † | 153 (8.7%) † |
| MSFT | 97 (5.5%) | 101 (5.8%) | 151 (8.6%) † | 192 (10.9%) † | 179 (10.2%) † |
| NVDA | 86 (4.9%) | 78 (4.4%) | 138 (7.9%) † | 194 (11.0%) † | 179 (10.2%) † |
| TSLA | 89 (5.1%) | 75 (4.3%) | 146 (8.3%) † | 181 (10.3%) † | 178 (10.1%) † |
| XOM | 100 (5.7%) | 97 (5.5%) | 146 (8.3%) † | 188 (10.7%) † | 190 (10.8%) † |
| ^GSPC | 108 (6.2%) † | 110 (6.3%) † | 166 (9.5%) † | 232 (13.2%) † | 222 (12.6%) † |
| **AVERAGE** | 95.4 | 91.9 | 144.4 | 191.0 | 182.1 |
| **Kupiec rejects (/7)** | 1 | 1 | 7 | 7 | 7 |

**99% VaR** — expected 17.6 breaches / 1756 sessions

| ticker | ewma_094 | garch_11 | har_rv | lgbm | lgbm_vix |
|---|---|---|---|---|---|
| AAPL | 40 (2.3%) † | 31 (1.8%) † | 49 (2.8%) † | 95 (5.4%) † | 95 (5.4%) † |
| JPM | 44 (2.5%) † | 38 (2.2%) † | 56 (3.2%) † | 88 (5.0%) † | 79 (4.5%) † |
| MSFT | 33 (1.9%) † | 38 (2.2%) † | 64 (3.6%) † | 100 (5.7%) † | 92 (5.2%) † |
| NVDA | 23 (1.3%) | 16 (0.9%) | 54 (3.1%) † | 85 (4.8%) † | 73 (4.2%) † |
| TSLA | 30 (1.7%) † | 29 (1.7%) † | 62 (3.5%) † | 101 (5.8%) † | 91 (5.2%) † |
| XOM | 37 (2.1%) † | 36 (2.1%) † | 60 (3.4%) † | 90 (5.1%) † | 96 (5.5%) † |
| ^GSPC | 42 (2.4%) † | 40 (2.3%) † | 76 (4.3%) † | 121 (6.9%) † | 117 (6.7%) † |
| **AVERAGE** | 35.6 | 32.6 | 60.1 | 97.1 | 91.9 |
| **Kupiec rejects (/7)** | 6 | 6 | 7 | 7 | 7 |

**Calibrated GK-target variants** (session-range variance rescaled to close-to-close by the walk-forward, training-only ratio c = mean(r^2)/mean(gk_var)).

*95% VaR — expected 87.8 breaches*

| ticker | har_rv_cal | lgbm_cal | lgbm_vix_cal |
|---|---|---|---|
| AAPL | 65 (3.7%) † | 105 (6.0%) | 106 (6.0%) |
| JPM | 73 (4.2%) | 102 (5.8%) | 96 (5.5%) |
| MSFT | 89 (5.1%) | 125 (7.1%) † | 122 (6.9%) † |
| NVDA | 70 (4.0%) † | 98 (5.6%) | 92 (5.2%) |
| TSLA | 81 (4.6%) | 118 (6.7%) † | 112 (6.4%) † |
| XOM | 96 (5.5%) | 124 (7.1%) † | 122 (6.9%) † |
| ^GSPC | 78 (4.4%) | 118 (6.7%) † | 116 (6.6%) † |
| **AVERAGE** | 78.9 | 112.9 | 109.4 |
| **Kupiec rejects (/7)** | 2 | 4 | 4 |

*99% VaR — expected 17.6 breaches*

| ticker | har_rv_cal | lgbm_cal | lgbm_vix_cal |
|---|---|---|---|
| AAPL | 24 (1.4%) | 42 (2.4%) † | 40 (2.3%) † |
| JPM | 31 (1.8%) † | 47 (2.7%) † | 41 (2.3%) † |
| MSFT | 31 (1.8%) † | 55 (3.1%) † | 49 (2.8%) † |
| NVDA | 14 (0.8%) | 30 (1.7%) † | 30 (1.7%) † |
| TSLA | 25 (1.4%) | 40 (2.3%) † | 44 (2.5%) † |
| XOM | 30 (1.7%) † | 46 (2.6%) † | 58 (3.3%) † |
| ^GSPC | 29 (1.7%) † | 60 (3.4%) † | 53 (3.0%) † |
| **AVERAGE** | 26.3 | 45.7 | 45.0 |
| **Kupiec rejects (/7)** | 4 | 7 | 7 |
<!-- VAR:END -->

**Limitations.** The VaR is parametric-normal, so it structurally cannot capture the
fat tails and volatility-of-volatility of real equity returns; at 99% especially, the
normal quantile sits inside the true tail, so under-coverage there is expected by
construction and is *measured*, not hidden, in the table above. A Student-t innovation
(heavier tails, one extra degree-of-freedom parameter) is the natural next step and
remains a documented stretch goal. Kupiec's POF test also has **low power at 99%** with
n ≈ 1,756 (only ~17.6 expected breaches): a non-rejection there is weak evidence of
correct coverage, not proof — 99% p-values are read with that caveat.

## Ablation results (v2)

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
uv run python -m volrisk.ingest.backfill        # OHLCV since 2016-07-11 (fixed inception) -> data/raw/*.parquet
uv run python -m volrisk.db.migrate             # apply db/migrations/*.sql (tracked, idempotent)
uv run python -m volrisk.db.load_raw            # upsert parquet -> raw.daily_bars (re-run: net 0)
uv run python -m volrisk.transform.cleaning     # calendar-align -> clean.daily_bars + gap report
uv run python -m volrisk.features.build         # window functions -> features.daily_features
uv run python -m volrisk.features.crosscheck    # SQL vs pandas recomputation, per ticker
uv run python -m volrisk.models.baselines       # walk-forward EWMA + GARCH -> forecasts schema
uv run python -m volrisk.models.feature_models  # walk-forward HAR-RV + LightGBM (+VIX variant)
uv run python -m volrisk.evaluate.ablation --write-readme   # QLIKE/RMSE tables -> DB + README
uv run python -m volrisk.risk.backtest --write-readme       # VaR + Kupiec coverage -> DB + README
uv run --env-file .env pytest                   # includes DB integration tests
```

(The two `--write-readme` flags regenerate the marked result sections below; omit them
to store results in Postgres only.)

Every load stage upserts on the natural key `(ticker, trade_date)`, so re-running any
stage is idempotent; recent bars are revisable by design (a fetch during market hours
lands an in-progress bar, which later runs revise to final values and the cleaning
stage excludes until its session has closed).

**Repo invariant: the modeling layer never sees an in-progress bar.** A bar reaches
`clean` — and everything downstream of it — only after its exchange session has closed.

Calendar note: ^VIX is CBOE-listed; the XNYS calendar is used as a proxy for the whole
US basket. That is a deliberate simplification — its artifacts (e.g. a phantom ^VIX bar
on a market holiday) are surfaced and excluded by the cleaning stage's gap report.

## Automation (Step 11)

A scheduled GitHub Actions job ([nightly.yml](.github/workflows/nightly.yml)) runs the
whole pipeline every trading day against a **Neon serverless Postgres**:
migrate → fetch (full anchored backfill via a **yfinance → Stooq fallback chain**) →
validate → load → clean → features → all forecasts → VaR backtest. One command runs it
anywhere: `uv run python -m volrisk.ingest.daily_update`.

**Schedule.** `30 22 * * 1-5` (22:30 UTC, Mon–Fri): NYSE closes 16:00 ET = 20:00 UTC
(EDT) / 21:00 UTC (EST), so one year-round cron line gives 1.5–2.5 h of slack for
Yahoo's final daily prints and finishes long before the next open. GitHub auto-disables
scheduled workflows after ~60 days without repo activity; it emails a warning first,
the workflow keeps a `workflow_dispatch` trigger for manual runs/re-enabling, and the
repo stays active through the roadmap.

**Canaries are exit codes.** Telescoping-identity failures, negative Garman–Klass
values, floored predictions, and GARCH fallback/unconverged refits each fail the job
after the summary prints — every data-quality invariant is re-proven nightly.

**Landing-zone semantics.** The cloud DB is the **system of record**. A runner's
parquet is deterministic staging, reconstructable from the fixed inception anchor; the
dev machine's `data/raw/` is the durable replay copy. A **monotonic guard** refuses any
fetch that would *shrink* a ticker's parquet (`--force` only after investigation);
guarded tickers fall back to a ~5-trading-day trailing-window fetch landed as dated
increment files under `data/raw/increments/` — the anchored zone is never overwritten.
Stooq fallback rows are adjusted-only (`close == adj_close` by policy) and flagged via
`raw.daily_bars.source`.

**Neon free tier** (verified from [neon.com/docs/introduction/plans](https://neon.com/docs/introduction/plans),
2026-07-17): $0/month — 100 CU-hours/project/month of compute (autoscaling up to 2 CU),
0.5 GB storage/project, 5 GB egress/month, scale-to-zero after 5 min (not disableable
on Free). **Known edge:** exhausting CU-hours or egress **suspends compute until the
next billing period** (or upgrade), and exceeding the storage cap blocks
storage-increasing writes. Our footprint — ~10 min of ≤2 CU compute/night ≈ a few
CU-hours/month and ~0.1 GB of data — sits at roughly 10% of the allowances, so the
cutoff is documented, not expected.

**Two databases.** Neon is production (the nightly job and Power BI read/write it);
local Postgres on port 5433 remains the dev/test DB. The cloud DB is seeded by
**replaying the pipeline from the anchor** (run `daily_update` once locally with
`DATABASE_URL` pointed at Neon) — which doubles as the landing-zone replayability
proof.
