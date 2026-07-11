-- Feature build: window functions over clean.daily_bars.
-- Full recompute, upserted on (ticker, trade_date) — idempotent by construction.
-- Every rolling statistic requires a FULL window (COUNT of non-null inputs must
-- equal the window length), so warm-up rows carry NULL instead of a silently
-- shortened-window value. All windows partition by ticker: no cross-ticker leakage.

INSERT INTO features.daily_features (
    ticker, trade_date, log_return,
    ret_lag_1, ret_lag_2, ret_lag_3, ret_lag_4, ret_lag_5,
    r2, park_var, gk_var,
    rvol_5, rvol_21, rvol_63,
    har_rv_w, har_rv_m,
    target_gk_var_next
)
SELECT
    ticker,
    trade_date,
    log_return,
    LAG(log_return, 1) OVER w_ord,
    LAG(log_return, 2) OVER w_ord,
    LAG(log_return, 3) OVER w_ord,
    LAG(log_return, 4) OVER w_ord,
    LAG(log_return, 5) OVER w_ord,
    r2,
    park_var,
    gk_var,
    CASE WHEN COUNT(r2) OVER w5  = 5  THEN sqrt(252 * AVG(r2) OVER w5)  END,
    CASE WHEN COUNT(r2) OVER w21 = 21 THEN sqrt(252 * AVG(r2) OVER w21) END,
    CASE WHEN COUNT(r2) OVER w63 = 63 THEN sqrt(252 * AVG(r2) OVER w63) END,
    CASE WHEN COUNT(gk_var) OVER w5  = 5  THEN AVG(gk_var) OVER w5  END,
    CASE WHEN COUNT(gk_var) OVER w22 = 22 THEN AVG(gk_var) OVER w22 END,
    LEAD(gk_var, 1) OVER w_ord
FROM (
    SELECT
        ticker,
        trade_date,
        log_return,
        log_return * log_return                          AS r2,
        ln(high / low) * ln(high / low) / (4 * ln(2))    AS park_var,
        0.5 * ln(high / low) * ln(high / low)
          - (2 * ln(2) - 1) * ln(close / open) * ln(close / open) AS gk_var
    FROM clean.daily_bars
) base
WINDOW
    w_ord AS (PARTITION BY ticker ORDER BY trade_date),
    w5    AS (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 4  PRECEDING AND CURRENT ROW),
    w21   AS (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW),
    w22   AS (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 21 PRECEDING AND CURRENT ROW),
    w63   AS (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW)
ON CONFLICT (ticker, trade_date) DO UPDATE SET
    log_return         = EXCLUDED.log_return,
    ret_lag_1          = EXCLUDED.ret_lag_1,
    ret_lag_2          = EXCLUDED.ret_lag_2,
    ret_lag_3          = EXCLUDED.ret_lag_3,
    ret_lag_4          = EXCLUDED.ret_lag_4,
    ret_lag_5          = EXCLUDED.ret_lag_5,
    r2                 = EXCLUDED.r2,
    park_var           = EXCLUDED.park_var,
    gk_var             = EXCLUDED.gk_var,
    rvol_5             = EXCLUDED.rvol_5,
    rvol_21            = EXCLUDED.rvol_21,
    rvol_63            = EXCLUDED.rvol_63,
    har_rv_w           = EXCLUDED.har_rv_w,
    har_rv_m           = EXCLUDED.har_rv_m,
    target_gk_var_next = EXCLUDED.target_gk_var_next,
    loaded_at          = now();
