-- 001_raw.sql — raw landing schema mirroring the canonical daily-bars contract.
-- The loader upserts on the natural key (ticker, trade_date); the CHECK
-- constraints are a second line of defense behind the pandera validation that
-- runs at ingest and again before load.

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.daily_bars (
    ticker      text             NOT NULL,
    trade_date  date             NOT NULL,
    open        double precision NOT NULL CHECK (open      > 0),
    high        double precision NOT NULL CHECK (high      > 0),
    low         double precision NOT NULL CHECK (low       > 0),
    close       double precision NOT NULL CHECK (close     > 0),
    adj_close   double precision NOT NULL CHECK (adj_close > 0),
    volume      bigint                    CHECK (volume   >= 0),
    loaded_at   timestamptz      NOT NULL DEFAULT now(),  -- operational timestamp, UTC
    PRIMARY KEY (ticker, trade_date),
    CHECK (high >= low),
    CHECK (open  BETWEEN low AND high),
    CHECK (close BETWEEN low AND high)
);
