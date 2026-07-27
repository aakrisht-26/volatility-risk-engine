-- 012_student_t.sql — Student-t VaR variants (Stretch 2).
--
-- The t variants are stored as ordinary coverage rows with model tags suffixed
-- '_t' (e.g. garch_11_t, har_rv_cal_t): same variance forecast as their
-- namesake, different quantile. That keeps Kupiec and Christoffersen working
-- unchanged and makes normal-vs-t a plain row comparison.
--
-- t_df records the MEDIAN degrees of freedom across the walk-forward refits
-- (df is re-estimated monthly on training data only, so it varies over the
-- window). NULL for normal-quantile rows — the column is what distinguishes
-- them in the data as well as in the tag.

ALTER TABLE forecasts.var_coverage
    ADD COLUMN t_df double precision;

ALTER TABLE forecasts.var_coverage
    ADD CONSTRAINT var_coverage_t_df_valid CHECK (t_df IS NULL OR t_df > 2);

DROP VIEW IF EXISTS dashboard.v_var_coverage;
CREATE VIEW dashboard.v_var_coverage AS
SELECT ticker, model, level, n_obs, expected_breaches, observed_breaches,
       breach_rate, kupiec_lr, kupiec_p,
       (kupiec_p < 0.05) AS kupiec_reject,
       n_00, n_01, n_10, n_11,
       lr_ind, p_ind, (p_ind < 0.05) AS independence_reject,
       lr_cc, p_cc, (p_cc < 0.05) AS conditional_coverage_reject,
       t_df, (t_df IS NOT NULL) AS is_student_t,
       eval_start, eval_end
FROM forecasts.var_coverage;
