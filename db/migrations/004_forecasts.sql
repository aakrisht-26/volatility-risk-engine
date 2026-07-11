-- 004_forecasts.sql — next-day variance forecasts from the model ladder.
--
-- UNITS ARE LOAD-BEARING. var_forecast is the DAILY variance of the LOG RETURN
-- in RETURN units — not percent² (arch's internal fitting scale), and not
-- annualized. Multiply by 252 for annualized variance; sqrt(252·x) for
-- annualized volatility. This matches features.gk_var, so forecast and
-- realized variance are directly comparable (QLIKE/RMSE at Step 8).
--
-- trade_date is the session BEING FORECAST: each row is produced using
-- information through the PREVIOUS session only (walk-forward, no leakage).

CREATE SCHEMA IF NOT EXISTS forecasts;

CREATE TABLE IF NOT EXISTS forecasts.daily_variance (
    ticker       text             NOT NULL,
    trade_date   date             NOT NULL,  -- the session being forecast
    model        text             NOT NULL,  -- model tag: 'ewma_094', 'garch_11', ...
    var_forecast double precision NOT NULL CHECK (var_forecast > 0),
    loaded_at    timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date, model)
);
