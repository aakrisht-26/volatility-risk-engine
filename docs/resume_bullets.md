# Resume bullets — volatility-risk-engine

Every number below is substantiated by the repo as of 2026-07-28 (regenerated README
tables, n = 1,762 walk-forward sessions/ticker, 2019-07-15..2026-07-17). Update numbers
from the README's generated tables if you regenerate before using these.

---

## Variant A — risk / quant framing (4 bullets)

- Built a volatility forecasting and VaR backtesting engine over 10 years of daily
  data for 8 US tickers: walk-forward EWMA, GARCH(1,1), log-log HAR-RV, and LightGBM
  forecasts evaluated with QLIKE over 1,762 out-of-sample sessions per ticker — HAR-RV
  on the Garman–Klass target beat GARCH by 27% on QLIKE (0.293 vs 0.401) and 29% on
  RMSE (10.8 vs 15.1 annualized-vol points).
- Backtested 1-day parametric VaR with Kupiec coverage tests against three
  pre-registered predictions (all confirmed): GARCH/EWMA sat within 0.3pp of nominal
  95% coverage, while a walk-forward variance calibration (training-only
  mean(r²)/mean(GK) ratio, no look-ahead) converted range-based forecasts to
  close-to-close and cut HAR-RV's 95% Kupiec rejections from 7/7 tickers to 2/7.
- Fixed the fat-tail gap the normal quantile leaves by fitting a variance-matched
  Student-t (df by MLE on standardized residuals, re-estimated at every monthly refit
  on training data only): average 99% breach rates fell 3.07% → 2.42% against 1%
  nominal and Kupiec rejections 51/56 → 42/56, with GARCH(1,1) improving 1.85% → 1.44%
  (6/7 → 3/7 tickers). The cost is reported alongside the gain — 95% coverage
  *worsens* (7.14% → 7.93%), because holding variance fixed moves distribution mass
  from the shoulders into the tails.
- Selected the production VaR model on measured coverage rather than preference:
  `har_rv_cal_t` is the best-calibrated combination in the project (95% breach rate
  **4.94%**, 2 of 7 tickers rejecting Kupiec; 99% **1.26%**, **1 of 7**), and GARCH(1,1)
  is retained as the published benchmark because it wins the independence dimension
  outright (0 of 7 rejections at both levels) — accuracy of the risk *level* and
  correctness of breach *timing* are different properties, and a desk would run both.

## Variant B — ML / modeling framing (2 bullets)

- Designed a leakage-proof walk-forward harness (expanding window, monthly refits,
  explicit no-leakage and prediction-alignment unit tests) for a four-model volatility
  ladder; diagnosed and fixed two opposite loss pathologies — near-zero level-space
  forecasts exploding QLIKE, and log-target-on-level-features exploding RMSE on vol
  spikes — by moving to Corsi's log-log HAR with a lognormal half-variance
  retransformation, lifting the best model to a 27% QLIKE improvement over GARCH(1,1).
- Demonstrated proxy-dependence of model rankings (range-based Garman–Klass vs
  close-to-close squared returns flip the winner), and resolved it for the risk
  application with a walk-forward, training-only calibration layer whose factor was
  largest for the index (2.0× vs 1.5–1.8× for single names), consistent with index
  ranges under-measuring dispersion.

## Variant C — data engineering framing (3 bullets)

- Built an end-to-end PostgreSQL analytics pipeline (raw → clean → features →
  forecasts → dashboard views) over ~20k daily bars with idempotent natural-key
  upserts at every stage, pandera batch validation plus 44 database CHECK constraints,
  and exchange-calendar alignment with an explicit partial-bar policy — re-running any
  stage adds zero duplicate rows.
- Wrote the feature layer in pure SQL window functions and pinned it with an
  independent pandas recomputation cross-check agreeing to ≤1e-15 per column; promoted
  data-quality invariants (telescoping return identity, provably-non-negative
  Garman–Klass, prediction floors, Student-t df clamping) to canaries that fail the
  pipeline loudly rather than degrading quietly.
- Automated the whole system as a scheduled GitHub Actions job (rehearsed end-to-end
  in ~5 minutes) with a two-provider fallback chain (yfinance → Stooq), a monotonic
  landing-zone guard against truncated fetches, provenance-tracked rows, and 149 tests
  running in CI against a PostgreSQL service container in two dependency
  configurations.

## Variant D — one-liner (for a summary section)

- Volatility forecasting & VaR engine (Python/PostgreSQL): four walk-forward models
  over 10y × 8 US tickers, best model 27% better QLIKE than GARCH(1,1); Kupiec- and
  Christoffersen-backtested VaR with normal and Student-t quantiles, model choice
  settled on measured coverage (best variant 4.94% / 1.26% against 5% / 1% nominal);
  scheduled-job automation, 149 tests, green CI.

## Variant E — methodology / scientific rigor (1 bullet, pairs with any variant)

- Ran the project as a pre-registered study: **ten predictions across three sets were
  committed to git before the corresponding results were computed** — the results
  blocks are verifiably empty in the registering commits — then graded in the README
  from generated output. Nine held; one was falsified and is published as a named
  error rather than quietly dropped; and one *technically confirmed* result was
  explicitly declined as supporting evidence because a power caveat registered
  alongside it explains the effect away. Anyone with the repository can check out the
  registering commit and grade the predictions themselves.

---

Notes for use:
- Automation wording is deliberately "scheduled job, rehearsed end-to-end" — the cron
  schedule is built but not yet activated. Upgrade to "running nightly since <date>"
  only after the first green scheduled run.
- "27% / 29%" derive from the averages 0.2930 vs 0.4009 (QLIKE) and 10.78 vs 15.11
  (RMSE) in the README's generated ablation table.
- "0.3pp of nominal" and "7/7 → 2/7" come from the generated VaR coverage block.
- The Student-t figures (3.07% → 2.42%, 51/56 → 42/56, 1.85% → 1.44%, 7.14% → 7.93%)
  come from the generated `STUDENTT` block. The 56 is 8 model variants × 7 tickers at
  one level; it is not a ticker count.
- Crown figures (4.94% / 2-of-7 and 1.26% / 1-of-7 for `har_rv_cal_t`; 0-of-7
  independence rejections for `garch_11`) come from the generated VaR and
  `INDEPENDENCE` blocks. The featured/benchmark split is the ruling recorded in
  CLAUDE.md (2026-07-22) — quote both models, never the crown alone.
- The Christoffersen direction split (27 of 48 rejections are clustering, 21 are
  anti-clustering, over 224 ticker × variant × level tests) comes from the
  `INDEPENDENCE` block, whose per-variant reject columns sum to those totals.
- "149 tests" is the suite with the optional `tracking` extra installed
  (`uv sync --extra tracking`); without it 147 run and 2 skip. Counted 2026-07-28.
- "44 CHECK constraints" counted from `pg_constraint` on 2026-07-28 across the raw (9),
  clean (9), features (5), and forecasts (21) schemas. Re-count before quoting if
  migrations change.
- "Ten predictions across three sets" = 3 VaR coverage (Step 9), 3 Christoffersen
  independence (stretch 1), 4 Student-t (stretch 2). The falsified one is Student-t
  prediction (ii); the declined-as-support one is Student-t prediction (iv).
- The index-calibration observation (2.045× vs 1.51–1.77×) is measured; its mechanism
  is a stated hypothesis — keep the "consistent with" phrasing.
