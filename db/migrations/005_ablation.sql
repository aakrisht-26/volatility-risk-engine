-- 005_ablation.sql — stored ablation metrics (Step 8).
-- qlike is unitless (lower is better). rmse_ann_vol_pct is RMSE expressed in
-- annualized-volatility percentage points for readability; the underlying
-- inputs are daily variances in return units like everything else.
-- Metrics are computed per (ticker, model) over the COMMON evaluation span —
-- dates where every compared model has a forecast for that ticker — so the
-- rows of one ticker are directly comparable.

CREATE TABLE IF NOT EXISTS forecasts.ablation_metrics (
    ticker           text             NOT NULL,
    model            text             NOT NULL,
    n_obs            integer          NOT NULL,
    qlike            double precision NOT NULL,
    rmse_ann_vol_pct double precision NOT NULL,
    eval_start       date             NOT NULL,
    eval_end         date             NOT NULL,
    computed_at      timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, model)
);
