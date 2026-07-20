-- 009_dashboard.sql — the Power BI surface (Step 12).
--
-- Power BI connects to the `dashboard` schema and NOTHING else; every visual
-- reads one of these views. Because the whole surface lives in the database,
-- re-pointing the dashboard from dev (localhost:5433) to Neon is config-only.
--
-- Breach FLAGS are computed here for charting; coverage STATISTICS (Kupiec)
-- come from the authoritative stored forecasts.var_coverage, not recomputed.
-- Units: *_ann_vol_pct are annualized volatility percentage points
-- (100*sqrt(252*variance)); var_threshold and log_return are daily log-return
-- units.

CREATE SCHEMA IF NOT EXISTS dashboard;

-- Page 2: forecast vs realized, in human units.
CREATE OR REPLACE VIEW dashboard.v_forecast_vs_realized AS
SELECT f.ticker,
       f.trade_date,
       f.model,
       f.var_forecast,
       x.gk_var                         AS realized_gk_var,
       sqrt(252 * f.var_forecast) * 100 AS forecast_ann_vol_pct,
       sqrt(252 * x.gk_var) * 100       AS realized_ann_vol_pct
FROM forecasts.daily_variance f
JOIN features.daily_features x USING (ticker, trade_date)
WHERE x.gk_var > 0;

-- Page 3 (breach tracker): full daily VaR series with breach flags, both levels.
CREATE OR REPLACE VIEW dashboard.v_var_daily AS
SELECT f.ticker,
       f.trade_date,
       f.model,
       z.level,
       c.log_return,
       z.zscore * sqrt(f.var_forecast)                   AS var_threshold,
       (c.log_return < -(z.zscore * sqrt(f.var_forecast))) AS breach
FROM forecasts.daily_variance f
JOIN clean.daily_bars c USING (ticker, trade_date)
CROSS JOIN (VALUES (95, 1.6448536269514722), (99, 2.3263478740408408)) AS z(level, zscore)
WHERE c.log_return IS NOT NULL;

-- Page 3 cards: authoritative coverage + Kupiec from the risk layer.
CREATE OR REPLACE VIEW dashboard.v_var_coverage AS
SELECT ticker, model, level, n_obs, expected_breaches, observed_breaches,
       breach_rate, kupiec_lr, kupiec_p,
       (kupiec_p < 0.05) AS kupiec_reject,
       eval_start, eval_end
FROM forecasts.var_coverage;

-- Page 4: the ablation table.
CREATE OR REPLACE VIEW dashboard.v_ablation AS
SELECT ticker, model, n_obs, qlike, rmse_ann_vol_pct, eval_start, eval_end
FROM forecasts.ablation_metrics;

-- Page 5: vol-regime timeline — rvol_21 ranked within each ticker's own history.
CREATE OR REPLACE VIEW dashboard.v_vol_regime AS
SELECT ticker,
       trade_date,
       rvol_21 * 100 AS ann_vol_pct_21d,
       pr            AS vol_percentile,
       CASE
           WHEN pr < 0.25 THEN 'calm'
           WHEN pr < 0.75 THEN 'normal'
           WHEN pr < 0.95 THEN 'elevated'
           ELSE 'stressed'
       END AS regime
FROM (
    SELECT ticker, trade_date, rvol_21,
           percent_rank() OVER (PARTITION BY ticker ORDER BY rvol_21) AS pr
    FROM features.daily_features
    WHERE rvol_21 IS NOT NULL
) ranked;

-- Page 1 cards: freshest forecast per ticker x model.
CREATE OR REPLACE VIEW dashboard.v_latest_forecast AS
SELECT DISTINCT ON (ticker, model)
       ticker, model, trade_date, var_forecast,
       sqrt(252 * var_forecast) * 100 AS ann_vol_pct
FROM forecasts.daily_variance
ORDER BY ticker, model, trade_date DESC;

-- The indexes deferred from the 2026-07-15 audit, settled alongside these
-- queries: import-mode refresh scans whole views (fine), but model-led and
-- date-led slicing — the two dashboard access paths — are not served by the
-- ticker-first primary keys.
CREATE INDEX IF NOT EXISTS daily_variance_model_date_idx
    ON forecasts.daily_variance (model, trade_date);
CREATE INDEX IF NOT EXISTS clean_daily_bars_date_idx
    ON clean.daily_bars (trade_date);
