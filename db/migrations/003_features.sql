-- 003_features.sql — model-ready features computed by SQL window functions.
-- Populated by src/volrisk/features/build_features.sql (full recompute, upsert
-- on the natural key). Units are documented per column: *_var columns are
-- DAILY variance of log returns; rvol_* are ANNUALIZED volatility (sqrt(252·)).

CREATE SCHEMA IF NOT EXISTS features;

CREATE TABLE IF NOT EXISTS features.daily_features (
    ticker              text             NOT NULL,
    trade_date          date             NOT NULL,
    log_return          double precision,  -- ln(adj_close_t / adj_close_{t-1}), from clean
    ret_lag_1           double precision,  -- log_return lagged 1..5 sessions
    ret_lag_2           double precision,
    ret_lag_3           double precision,
    ret_lag_4           double precision,
    ret_lag_5           double precision,
    r2                  double precision,  -- squared log return: noisy daily variance proxy (robustness check)
    park_var            double precision,  -- Parkinson daily variance: ln(H/L)^2 / (4 ln 2)
    gk_var              double precision,  -- Garman–Klass daily variance (primary RV proxy):
                                           --   0.5·ln(H/L)^2 − (2 ln 2 − 1)·ln(C/O)^2
    rvol_5              double precision,  -- annualized realized vol: sqrt(252 · avg(r2) over full window)
    rvol_21             double precision,
    rvol_63             double precision,
    har_rv_w            double precision,  -- HAR weekly component: 5-day mean of gk_var
    har_rv_m            double precision,  -- HAR monthly component: 22-day mean of gk_var
                                           -- (HAR daily component IS gk_var)
    target_gk_var_next  double precision,  -- next-day gk_var: the forecasting target
    loaded_at           timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);
