-- 010_live_forecasts.sql — the live next-session VaR row (Step-12 ruling).
--
-- A live row is the forecast FOR the next session after the last completed
-- one — the number a risk desk actually wants before tomorrow's open. It has
-- no realized outcome yet, so it is EXCLUDED from backtest/coverage/ablation:
-- structurally (no clean/features row exists at its date to join) AND
-- explicitly (loaders filter NOT is_live; pinned by tests). When the session
-- completes, the nightly walk-forward re-emits that date and the upsert
-- overwrites the row with is_live = false.

ALTER TABLE forecasts.daily_variance
    ADD COLUMN is_live boolean NOT NULL DEFAULT false;

-- Freshest forecast per ticker x model; live rows (strictly newest by date)
-- surface first for the Overview card, flagged so the card can label them.
-- DROP first: CREATE OR REPLACE cannot insert a column before ann_vol_pct.
DROP VIEW IF EXISTS dashboard.v_latest_forecast;
CREATE VIEW dashboard.v_latest_forecast AS
SELECT DISTINCT ON (ticker, model)
       ticker, model, trade_date, var_forecast, is_live,
       sqrt(252 * var_forecast) * 100 AS ann_vol_pct
FROM forecasts.daily_variance
ORDER BY ticker, model, trade_date DESC;
