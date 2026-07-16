-- 007_sanity_checks.sql — cheap invariants on the derived tables (audit fix 4).
--
-- Deliberately NOT constrained: features.gk_var and its derivatives
-- (har_rv_w, har_rv_m, target_gk_var_next). gk_var IS the data-quality canary —
-- it is provably non-negative on valid OHLC, so a negative value must be able
-- to LAND and fire the negative_gk counter rather than be rejected here.
-- ADD CONSTRAINT validates existing rows; a failure below means real data
-- violates an invariant and must be investigated, not forced.

ALTER TABLE features.daily_features
    ADD CONSTRAINT daily_features_r2_nonneg       CHECK (r2 >= 0),
    ADD CONSTRAINT daily_features_park_nonneg     CHECK (park_var >= 0),
    ADD CONSTRAINT daily_features_rvol5_nonneg    CHECK (rvol_5 >= 0),
    ADD CONSTRAINT daily_features_rvol21_nonneg   CHECK (rvol_21 >= 0),
    ADD CONSTRAINT daily_features_rvol63_nonneg   CHECK (rvol_63 >= 0);

ALTER TABLE forecasts.ablation_metrics
    ADD CONSTRAINT ablation_n_obs_positive        CHECK (n_obs > 0),
    ADD CONSTRAINT ablation_qlike_nonneg          CHECK (qlike >= 0),
    ADD CONSTRAINT ablation_rmse_nonneg           CHECK (rmse_ann_vol_pct >= 0),
    ADD CONSTRAINT ablation_span_ordered          CHECK (eval_start <= eval_end);

ALTER TABLE forecasts.var_coverage
    ADD CONSTRAINT var_coverage_level_valid       CHECK (level IN (95, 99)),
    ADD CONSTRAINT var_coverage_n_obs_positive    CHECK (n_obs > 0),
    ADD CONSTRAINT var_coverage_expected_positive CHECK (expected_breaches > 0),
    ADD CONSTRAINT var_coverage_observed_bounded  CHECK (observed_breaches BETWEEN 0 AND n_obs),
    ADD CONSTRAINT var_coverage_rate_bounded      CHECK (breach_rate >= 0 AND breach_rate <= 1),
    ADD CONSTRAINT var_coverage_lr_nonneg         CHECK (kupiec_lr >= 0),
    ADD CONSTRAINT var_coverage_p_bounded         CHECK (kupiec_p >= 0 AND kupiec_p <= 1),
    ADD CONSTRAINT var_coverage_span_ordered      CHECK (eval_start <= eval_end);

ALTER TABLE forecasts.var_breaches
    ADD CONSTRAINT var_breaches_level_valid       CHECK (level IN (95, 99)),
    ADD CONSTRAINT var_breaches_threshold_positive CHECK (var_threshold > 0),
    -- The defining invariant of this sparse table: a stored row IS a breach.
    ADD CONSTRAINT var_breaches_is_a_breach       CHECK (log_return < -var_threshold);
