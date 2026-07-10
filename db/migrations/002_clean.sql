-- 002_clean.sql — calendar-aligned, completed-session daily bars with log returns.
-- Populated by volrisk.transform.cleaning: bars aligned against the ticker's
-- exchange calendar (XNYS for the US basket), partial same-day bars and
-- non-session rows excluded per the Step-5 policy, log returns computed on
-- adjusted close over consecutive available sessions.

CREATE SCHEMA IF NOT EXISTS clean;

CREATE TABLE IF NOT EXISTS clean.daily_bars (
    ticker      text             NOT NULL,
    trade_date  date             NOT NULL,
    open        double precision NOT NULL CHECK (open      > 0),
    high        double precision NOT NULL CHECK (high      > 0),
    low         double precision NOT NULL CHECK (low       > 0),
    close       double precision NOT NULL CHECK (close     > 0),
    adj_close   double precision NOT NULL CHECK (adj_close > 0),
    volume      bigint                    CHECK (volume   >= 0),
    log_return  double precision,  -- ln(adj_close_t / adj_close_{t-1}); NULL on each ticker's first row
    loaded_at   timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date),
    CHECK (high >= low),
    CHECK (open  BETWEEN low AND high),
    CHECK (close BETWEEN low AND high)
);
