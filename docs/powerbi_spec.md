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
  red when > 4 — data is stale), and per-candidate **latest ann. vol** — two
  multi-row cards from `v_latest_forecast` filtered (visual-level) to
  `model = garch_11` and `model = har_rv_cal`, fields ticker + `ann_vol_pct`.
- Slicer: none (this page is glanceable).
- Text box: one-line positioning statement (risk analytics, not signals).

### Page 2 — Forecast vs Realized
- Line chart: Axis `DimDate[Date]`, Values **Forecast Vol %** and **Realized Vol %**.
- Slicers: `DimTicker[ticker]` (single-select dropdown), `DimModel[model]`
  (single-select; default `har_rv`).
- Note the visual reads annualized-vol units — human-readable per the repo's
  units discipline.

### Page 3 — VaR Breach Tracker ("the crown page")
Slicers at top: `DimTicker[ticker]` (single-select), `v_var_daily[level]`
(buttons: 95 / 99). Two identical visual columns, LEFT filtered
(visual-level) to `model = garch_11`, RIGHT to `model = har_rv_cal`:

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

**Crown scorecard** (computed 2026-07-20 from these exact views, n = 1,756/ticker):

| | garch_11 | har_rv_cal |
|---|---|---|
| 95% avg breach rate (nominal 5%) | 5.21% — 1/7 rejects | 4.48% — 2/7 rejects (over-covers) |
| 99% avg breach rate (nominal 1%) | 1.85% — 6/7 rejects | **1.49% — 4/7 rejects (least-bad)** |
| Avg daily VaR (capital) 95 / 99 | 3.514% / 4.970% | 3.532% / 4.995% (dead heat) |
| Accuracy context | wins r² proxy QLIKE | wins GK-proxy QLIKE + RMSE everywhere |

**Decision procedure on the page**: with capital efficiency tied, the crown
hangs on which coverage failure you prefer — garch_11 is *exact* at 95% but
worst-in-class at 99%; har_rv_cal is slightly conservative at 95% (the right
side to miss, for risk) and materially closer to nominal at 99%.
**Recommendation: har_rv_cal**, with garch_11 as the stated benchmark. The
crown is Aakrisht's call after seeing the page render.

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

- **Freshest-forecast semantics**: the newest row per model is the forecast FOR
  the most recent completed session (used in backtesting). A true
  next-session-ahead row (t+1 "live" VaR before tomorrow's open) is a small
  runner extension, deliberately not built yet — flag if wanted for the
  Overview page.
- The `_cal` variants are persisted as first-class forecast rows nightly (the
  ablation's QLIKE/RMSE tables deliberately exclude them; coverage includes them).
- Deferred-index decision (2026-07-15 audit) is settled in migration 009:
  `daily_variance(model, trade_date)` and `clean.daily_bars(trade_date)` — the
  two dashboard access paths the ticker-first PKs don't serve.
