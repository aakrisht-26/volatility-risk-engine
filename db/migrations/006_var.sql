-- 006_var.sql — 1-day parametric VaR coverage backtesting (Step 9).
--
-- VaR is parametric-normal, zero-mean: VaR_alpha(d) = z_alpha * sqrt(var_forecast(d)),
-- where var_forecast is the model's DAILY variance in RETURN units (so sigma and the
-- VaR threshold are both in log-return units, directly comparable to the realized
-- close-to-close log return). Breach convention (session d): the realized log return
-- r_d < -VaR_alpha(d) — the long-position loss tail.

CREATE TABLE IF NOT EXISTS forecasts.var_coverage (
    ticker            text             NOT NULL,
    model             text             NOT NULL,  -- includes _cal variants when calibration is built
    level             integer          NOT NULL,  -- VaR confidence level: 95 or 99
    n_obs             integer          NOT NULL,
    expected_breaches double precision NOT NULL,   -- (1 - level/100) * n_obs
    observed_breaches integer          NOT NULL,
    breach_rate       double precision NOT NULL,   -- observed / n_obs
    kupiec_lr         double precision NOT NULL,   -- POF likelihood-ratio statistic
    kupiec_p          double precision NOT NULL,   -- chi-square(1) p-value; low => reject correct coverage
    eval_start        date             NOT NULL,
    eval_end          date             NOT NULL,
    computed_at       timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, model, level)
);

-- Breach events (indicator = 1 dates): the breach series in sparse form. The full
-- 0/1 series is these rows over the eval window recorded in var_coverage; absent
-- dates are non-breaches. Feeds the Step-12 Power BI breach tracker.
CREATE TABLE IF NOT EXISTS forecasts.var_breaches (
    ticker        text             NOT NULL,
    model         text             NOT NULL,
    level         integer          NOT NULL,
    trade_date    date             NOT NULL,  -- session that breached
    log_return    double precision NOT NULL,  -- realized close-to-close log return that day
    var_threshold double precision NOT NULL,  -- VaR magnitude (positive); breach = log_return < -var_threshold
    PRIMARY KEY (ticker, model, level, trade_date)
);
