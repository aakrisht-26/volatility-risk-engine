-- 011_christoffersen.sql — independence and conditional-coverage statistics.
--
-- Kupiec (already stored as kupiec_lr/kupiec_p) tests the breach RATE.
-- Christoffersen's LR_ind tests whether breaches CLUSTER (first-order Markov
-- on the breach series), and LR_cc = LR_uc + LR_ind tests both jointly.
-- Columns live beside the coverage row they describe: same grain
-- (ticker, model, level), same evaluation window.
--
-- Transition counts are stored raw so a reader can audit the statistic:
-- n_11 (breach following a breach) is the clustering signal, and at 99% it is
-- expected to be near zero — which is exactly why the test is underpowered
-- there. Nullable: rows written before this migration have no values.

ALTER TABLE forecasts.var_coverage
    ADD COLUMN n_00     integer,
    ADD COLUMN n_01     integer,
    ADD COLUMN n_10     integer,
    ADD COLUMN n_11     integer,
    ADD COLUMN lr_ind   double precision,   -- Christoffersen independence, chi2(1)
    ADD COLUMN p_ind    double precision,   -- low => breaches cluster
    ADD COLUMN lr_cc    double precision,   -- conditional coverage, chi2(2)
    ADD COLUMN p_cc     double precision;

ALTER TABLE forecasts.var_coverage
    ADD CONSTRAINT var_coverage_lr_ind_nonneg CHECK (lr_ind IS NULL OR lr_ind >= 0),
    ADD CONSTRAINT var_coverage_lr_cc_nonneg  CHECK (lr_cc  IS NULL OR lr_cc  >= 0),
    ADD CONSTRAINT var_coverage_p_ind_bounded CHECK (p_ind IS NULL OR (p_ind >= 0 AND p_ind <= 1)),
    ADD CONSTRAINT var_coverage_p_cc_bounded  CHECK (p_cc  IS NULL OR (p_cc  >= 0 AND p_cc  <= 1));

-- Dashboard surface: expose the new statistics next to the Kupiec ones so the
-- Step-12 breach page can show clustering without a schema change later.
-- DROP first: CREATE OR REPLACE cannot insert columns before existing ones.
DROP VIEW IF EXISTS dashboard.v_var_coverage;
CREATE VIEW dashboard.v_var_coverage AS
SELECT ticker, model, level, n_obs, expected_breaches, observed_breaches,
       breach_rate, kupiec_lr, kupiec_p,
       (kupiec_p < 0.05) AS kupiec_reject,
       n_00, n_01, n_10, n_11,
       lr_ind, p_ind, (p_ind < 0.05) AS independence_reject,
       lr_cc, p_cc, (p_cc < 0.05) AS conditional_coverage_reject,
       eval_start, eval_end
FROM forecasts.var_coverage;
