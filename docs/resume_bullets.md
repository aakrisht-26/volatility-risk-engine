# Resume bullets — volatility-risk-engine

Every number below is substantiated by the repo as of 2026-07-20 (regenerated README
tables, n = 1,762 walk-forward sessions/ticker, 2019-07-15..2026-07-17). Update numbers
from the README's generated tables if you regenerate before using these.

---

## Variant A — risk / quant framing (2 bullets)

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
  upserts at every stage, pandera batch validation plus 39 database CHECK constraints,
  and exchange-calendar alignment with an explicit partial-bar policy — re-running any
  stage adds zero duplicate rows.
- Wrote the feature layer in pure SQL window functions and pinned it with an
  independent pandas recomputation cross-check agreeing to ≤1e-15 per column; promoted
  data-quality invariants (telescoping return identity, provably-non-negative
  Garman–Klass, prediction floors) to canaries that fail the pipeline loudly.
- Automated the whole system as a scheduled GitHub Actions job (rehearsed end-to-end
  in ~5 minutes) with a two-provider fallback chain (yfinance → Stooq), a monotonic
  landing-zone guard against truncated fetches, provenance-tracked rows, and 118 tests
  running in CI against a PostgreSQL service container.

## Variant D — one-liner (for a summary section)

- Volatility forecasting & VaR engine (Python/PostgreSQL): four walk-forward models
  over 10y × 8 US tickers, best model 27% better QLIKE than GARCH(1,1); Kupiec-backtested
  VaR with pre-registered, confirmed coverage predictions; scheduled-job automation,
  118 tests, green CI.

---

Notes for use:
- Automation wording is deliberately "scheduled job, rehearsed end-to-end" — the cron
  schedule is built but not yet activated. Upgrade to "running nightly since <date>"
  only after the first green scheduled run.
- "27% / 29%" derive from the averages 0.2930 vs 0.4009 (QLIKE) and 10.78 vs 15.11
  (RMSE) in the README's generated ablation table.
- "0.3pp of nominal" and "7/7 → 2/7" come from the generated VaR coverage block.
- "39 CHECK constraints" counted from `pg_constraint` on 2026-07-20 across the raw,
  clean, features, and forecasts schemas (9 raw + 9 clean + migration 007's 20 + the
  forecasts positivity check). Re-count before quoting if migrations change.
- The index-calibration observation (2.045× vs 1.51–1.77×) is measured; its mechanism
  is a stated hypothesis — keep the "consistent with" phrasing.
