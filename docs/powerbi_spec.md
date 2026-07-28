# Step 12 — Power BI dashboard spec (build-by-hand guide)

Everything Power BI touches lives in the **`dashboard` schema** (migration 009):
six views, nothing else. Re-pointing dev → Neon is config-only (two Power Query
parameters). Aakrisht builds the PBIX in Power BI Desktop from this spec.

---

## 0. Connection & refresh

1. **Get data → PostgreSQL database.**
2. First create two **Power Query parameters** (Manage Parameters → New):
   - `PgServer` (Text) = `localhost:5433`
   - `PgDatabase` (Text) = `volrisk`
3. Connect using the parameters (Server = `PgServer`, Database = `PgDatabase`),
   **Import** mode. Credentials: the `volrisk` role (Database auth, not Windows).
4. Select exactly the six `dashboard.*` views below.
5. **Re-point to Neon later**: edit `PgServer` to the Neon host, `PgDatabase` to
   `volrisk`, set credentials, tick *Encrypt connection*. Nothing else changes.
6. Refresh = Home → Refresh after each nightly run (Desktop). Scheduled refresh
   needs the Power BI Service + (for Neon) no gateway; out of scope until Step 11
   activation.

## 1. Data model

| Table (view) | Grain | Feeds |
|---|---|---|
| `v_forecast_vs_realized` | ticker × date × model | Page 2 |
| `v_var_daily` | ticker × date × model × level | Page 3 charts |
| `v_var_coverage` | ticker × model × level | Page 3 cards |
| `v_ablation` | ticker × model | Page 4 |
| `v_vol_regime` | ticker × date | Page 5 |
| `v_latest_forecast` | ticker × model | Page 1 cards |

**Power Query steps** (Transform Data):
- `v_var_daily`: add column `breach_int = Number.From([breach])` (Whole Number);
  add `neg_var_threshold = -[var_threshold] * 100` and `log_return_pct = [log_return] * 100` (Decimal).
- `v_var_coverage`: add `reject_int = Number.From([kupiec_reject])`.
- Create `DimTicker` = Reference `v_var_daily` → keep `ticker` → Remove duplicates.
- Create `DimModel` = Reference `v_var_daily` → keep `model` → Remove duplicates.

**DAX calculated table** (Modeling → New table), marked as Date table:

```DAX
DimDate = CALENDAR ( DATE ( 2016, 7, 11 ), TODAY () )
```

**Relationships** (single-direction, Many-to-one):
- `DimDate[Date]` → `v_forecast_vs_realized[trade_date]`, `v_var_daily[trade_date]`, `v_vol_regime[trade_date]`
- `DimTicker[ticker]` → same three fact tables + `v_var_coverage[ticker]`, `v_ablation[ticker]`, `v_latest_forecast[ticker]`
- `DimModel[model]` → `v_var_daily[model]`, `v_forecast_vs_realized[model]`, `v_var_coverage[model]`, `v_ablation[model]`, `v_latest_forecast[model]`

## 2. Measures (one dedicated `_Measures` table)

```DAX
Total Days = COUNTROWS ( v_var_daily )

Breaches = SUM ( v_var_daily[breach_int] )

Expected Rate % =
AVERAGEX ( VALUES ( v_var_daily[level] ), 100 - v_var_daily[level] )

Expected Breaches = DIVIDE ( [Total Days] * [Expected Rate %], 100 )

Breach Rate % = DIVIDE ( [Breaches], [Total Days] ) * 100

Coverage Gap = [Breaches] - [Expected Breaches]

Kupiec p = MIN ( v_var_coverage[kupiec_p] )

Kupiec Rejections = SUM ( v_var_coverage[reject_int] )

Avg Daily VaR % = AVERAGE ( v_var_daily[var_threshold] ) * 100

Cumulative Breaches =
CALCULATE (
    [Breaches],
    FILTER ( ALL ( DimDate[Date] ), DimDate[Date] <= MAX ( DimDate[Date] ) )
)

Cumulative Expected =
CALCULATE (
    [Expected Breaches],
    FILTER ( ALL ( DimDate[Date] ), DimDate[Date] <= MAX ( DimDate[Date] ) )
)

Forecast Vol % = AVERAGE ( v_forecast_vs_realized[forecast_ann_vol_pct] )

Realized Vol % = AVERAGE ( v_forecast_vs_realized[realized_ann_vol_pct] )

QLIKE = AVERAGE ( v_ablation[qlike] )

RMSE (vol pts) = AVERAGE ( v_ablation[rmse_ann_vol_pct] )

Latest Data Date = MAX ( v_latest_forecast[trade_date] )

Data Age (days) = DATEDIFF ( [Latest Data Date], TODAY (), DAY )
```

Formatting: `Kupiec p` 3 decimals; rates/vols 1–2 decimals; dates dd-mmm-yyyy.

## 3. Pages

### Page 1 — Overview
- Cards: **Latest Data Date**, **Data Age (days)** (conditional format: background
  red when > 4 — data is stale), and the **live next-session VaR card** for the
  FEATURED model: multi-row card from `v_latest_forecast` filtered (visual-level)
  to `model = har_rv_cal_t` and `is_live = True`, fields ticker + `ann_vol_pct` —
  title it "Next-session vol forecast (live)". A second, smaller card shows the
  benchmark (`model = garch_11`, `is_live = True`).
- Live rows are the forecast FOR the next session after the last completed one
  (flag `is_live`); they are excluded from every backtest/evaluation table by
  construction and by tested filters.
- Slicer: none (this page is glanceable).
- Text box: one-line positioning statement (risk analytics, not signals).

### Page 2 — Forecast vs Realized
- Line chart: Axis `DimDate[Date]`, Values **Forecast Vol %** and **Realized Vol %**.
- Slicers: `DimTicker[ticker]` (single-select dropdown), `DimModel[model]`
  (single-select; default `har_rv`).
- Note the visual reads annualized-vol units — human-readable per the repo's
  units discipline.

### Page 3 — VaR Breach Tracker
Slicers at top: `DimTicker[ticker]` (single-select, **default JPM** — the ticker
whose breaches actually show dependence), `v_var_daily[level]` (buttons: 95 /
99). Two identical visual columns, LEFT filtered (visual-level) to
`model = garch_11` (benchmark), RIGHT to `model = har_rv_cal_t` (featured):

1. **Return-vs-VaR band** — line chart: Axis `DimDate[Date]`; Values
   `log_return_pct` (grey, thin) and `neg_var_threshold` (candidate color).
   Breach markers: add scatter overlay via combo — simplest faithful approach:
   a second Values series `Breach Return = IF ( MAX(v_var_daily[breach_int]) = 1,
   MAX(v_var_daily[log_return_pct]) )` shown as markers-only (line transparency
   100%, markers on, red).
2. **Cumulative breaches vs expected** — line chart: **Cumulative Breaches**
   (candidate color) vs **Cumulative Expected** (dashed grey). A well-calibrated
   model tracks the grey line.
3. **Coverage card row** (from `v_var_coverage`): Breaches, Expected Breaches,
   Breach Rate %, **Kupiec p** — conditional format Kupiec p red when < 0.05.
4. **Avg Daily VaR %** card — the capital-efficiency number.

**Crown scorecard** (computed 2026-07-22 from these exact views, n = 1,762/ticker):

| | garch_11 (benchmark) | har_rv_cal_t (featured) |
|---|---|---|
| 95% avg breach rate (nominal 5%) | 5.21% — 1/7 rejects | **4.94% — 2/7 rejects** |
| 99% avg breach rate (nominal 1%) | 1.85% — 6/7 rejects | **1.26% — 1/7 rejects** |
| Independence (LR_ind rejects) | **0/7 at both levels** | 2/7 at both levels |
| Accuracy context | wins r² proxy QLIKE | wins GK-proxy QLIKE + RMSE everywhere |

Note for the page: `har_rv_cal_t` shares `har_rv_cal`'s **variance** forecast
exactly — only the quantile differs — so the Overview vol number is identical
between them; the tag matters for the VaR threshold, not the vol.

**Crown status — SETTLED 2026-07-22: `har_rv_cal_t` is the FEATURED model
across the dashboard; `garch_11` is the stated BENCHMARK on every page where
both appear** — label them exactly that way in visual titles.

Rationale to reproduce in a text box on this page: har_rv_cal_t is the
best-calibrated model in the project (95% breach rate 4.94%, 2/7 Kupiec
rejects; 99% 1.26%, 1/7), and coverage is the primary VaR criterion because it
is what the risk number claims to deliver. **garch_11 wins independence
outright (0/7 at both levels vs 2/7), and that advantage must be stated
wherever the crown is stated** — bursty breaches are worse than evenly spaced
ones, because losses compound faster than capital replenishes. Honest framing:
*har_rv_cal_t for accuracy of the risk level, garch_11 for timing behaviour; a
production desk would plausibly run both.*

**The crown procedure was revised on 2026-07-21 — this page no longer decides
it.** The original plan (eyeball whether har_rv_cal's breaches cluster) was
superseded by the Christoffersen test built to formalize it, which showed the
eyeball underdetermined: about half of all independence rejections are
*anti*-clustering (too-regular spacing, which looks healthy to the eye), and the
genuine clustering is ticker-specific (JPM and XOM only), so whichever ticker
was on screen would have decided the question. The crown now rests on the
stored statistics — direction-split independence (genuine clustering only),
Kupiec coverage, capital cost — and **this page illustrates them rather than
supplying the verdict.**

Accordingly, add a fourth visual to each column and default the page slicer to
**JPM**, the ticker where the clustering signal actually lives:

4. **Independence verdict card row** (from `v_var_coverage`, now carrying
   Christoffersen columns): `n_11` (breaches following a breach), **LR_ind
   p-value** with conditional formatting red when < 0.05, and **LR_cc
   p-value**. Add a DAX measure that names the *direction* so a reader never
   mistakes anti-clustering for clustering:

```DAX
Breach Dependence =
VAR Pi01 =
    DIVIDE ( SUM ( v_var_coverage[n_01] ), SUM ( v_var_coverage[n_00] ) + SUM ( v_var_coverage[n_01] ) )
VAR Pi11 =
    DIVIDE ( SUM ( v_var_coverage[n_11] ), SUM ( v_var_coverage[n_10] ) + SUM ( v_var_coverage[n_11] ) )
RETURN
    SWITCH (
        TRUE (),
        MIN ( v_var_coverage[p_ind] ) >= 0.05, "independent",
        Pi11 > Pi01, "CLUSTERED (risk failure)",
        "over-regular (not a risk failure)"
    )
```

Show this measure as a card beside the p-value; only "CLUSTERED" counts against
a model.

### Page 4 — Model Ablation
- Matrix: Rows `ticker`, Columns `model`, Values **QLIKE**; second matrix for
  **RMSE (vol pts)**. Conditional formatting → Background color scale per row
  (Power BI cannot bold the row minimum natively; a white→green scale reversed
  so lowest = deepest green is the honest equivalent).
- Footnote text box: the QLIKE formula + units, copied from the README footnote.

### Page 5 — Vol Regime Timeline
- Area/line chart per ticker (small multiples: `ticker`): Axis `DimDate[Date]`,
  Values `ann_vol_pct_21d`, Legend `regime` (calm/normal/elevated/stressed —
  colors green/grey/amber/red).
- Regime is each ticker's own historical percentile of 21-day realized vol
  (<25% calm, <75% normal, <95% elevated, else stressed) — computed in the view.

## 4. Notes & known semantics

- **Freshest-forecast semantics**: every model now also emits a **live
  next-session row** (`is_live = true`) — the forecast for the first session
  after the last completed one, including calibrated `_cal` variants (base live
  row × the month's training-only factor). Live rows surface in
  `v_latest_forecast` for the Overview card and are excluded from
  backtest/coverage/ablation both structurally (no realized outcome to join)
  and by explicit `NOT is_live` filters, pinned by tests. When the session
  completes, the nightly walk-forward re-emits the date and the upsert flips
  the flag to false.
- The `_cal` variants are persisted as first-class forecast rows nightly (the
  ablation's QLIKE/RMSE tables deliberately exclude them; coverage includes them).
- Deferred-index decision (2026-07-15 audit) is settled in migration 009:
  `daily_variance(model, trade_date)` and `clean.daily_bars(trade_date)` — the
  two dashboard access paths the ticker-first PKs don't serve.
