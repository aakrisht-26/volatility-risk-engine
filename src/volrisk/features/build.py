"""Build the features layer via SQL window functions.

Usage::

    uv run python -m volrisk.features.build

Executes ``build_features.sql``: a full recompute from clean.daily_bars,
upserted on (ticker, trade_date) — idempotent. Full recompute over the Phase-1
basket is instantaneous; incremental optimization is Step-11 scope.

The printed ``negative_gk`` column is a canary, not a statistic: Garman-Klass
is provably non-negative on valid OHLC (low ≤ open,close ≤ high implies
|ln(C/O)| ≤ ln(H/L)), so any non-zero count means invalid bars leaked past the
pandera and CHECK-constraint layers.
"""

from __future__ import annotations

import logging
from importlib import resources

from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine

logger = logging.getLogger(__name__)


def build_features(engine: Engine) -> int:
    """Run the feature-build SQL; returns the resulting features row count."""
    sql = resources.files("volrisk.features").joinpath("build_features.sql").read_text("utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM features.daily_features")).scalar_one()
    logger.info("features.daily_features holds %d rows", count)
    return count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    engine = get_engine()
    total = build_features(engine)

    with engine.connect() as conn:
        per_ticker = conn.execute(
            text(
                "SELECT ticker, count(*) AS rows, count(target_gk_var_next) AS targets,"
                " count(*) FILTER (WHERE gk_var < 0) AS negative_gk"
                " FROM features.daily_features GROUP BY ticker ORDER BY ticker"
            )
        ).all()

    print(f"\nfeatures.daily_features: {total} rows")
    print(f"{'ticker':>7} {'rows':>6} {'targets':>8} {'negative_gk':>12}")
    for ticker, rows, targets, negative_gk in per_ticker:
        print(f"{ticker:>7} {rows:>6} {targets:>8} {negative_gk:>12}")


if __name__ == "__main__":
    main()
