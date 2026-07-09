"""Load every parquet in the landing zone into Postgres ``raw.daily_bars``.

Usage::

    uv run python -m volrisk.db.load_raw [--raw-dir data/raw]

Idempotent by construction (upsert on the natural key): re-running adds zero
rows. Before/after table counts are printed to prove it on every run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from volrisk.db.engine import get_engine
from volrisk.db.loaders import raw_daily_bars_count, read_landing_parquet, upsert_daily_bars

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load raw parquet into raw.daily_bars.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    files = sorted(args.raw_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found under {args.raw_dir}")

    engine = get_engine()
    before = raw_daily_bars_count(engine)
    total = 0
    for path in files:
        df = read_landing_parquet(path)
        total += upsert_daily_bars(engine, df, context=path.name)
        logger.info("%s: upserted %d rows", path.name, len(df))
    after = raw_daily_bars_count(engine)

    print(f"\nraw.daily_bars before: {before:>7}")
    print(f"rows upserted:         {total:>7}")
    print(f"raw.daily_bars after:  {after:>7}")
    print(f"net new rows:          {after - before:>7}")


if __name__ == "__main__":
    main()
