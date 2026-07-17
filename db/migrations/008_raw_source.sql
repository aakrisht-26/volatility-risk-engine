-- 008_raw_source.sql — provider provenance on raw bars (Step 11).
--
-- 'yfinance' is the primary provider. 'stooq' rows come from the fallback
-- provider, whose data is ADJUSTED-ONLY: for those rows close == adj_close by
-- policy (see providers/stooq_provider.py), so consumers of the UNADJUSTED
-- close must filter on source. Returns computed from adj_close — the modeling
-- path — are provider-agnostic.
--
-- Deliberately no CHECK on the value: adding a provider should not require a
-- migration; the column is provenance metadata, not a contract.

ALTER TABLE raw.daily_bars
    ADD COLUMN source text NOT NULL DEFAULT 'yfinance';
