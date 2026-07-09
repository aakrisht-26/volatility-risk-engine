"""Apply plain versioned SQL migrations from db/migrations/ in filename order.

Applied versions are tracked in ``public.schema_migrations``, so re-running is
idempotent. Deliberately minimal — alembic is a later stretch goal per
CLAUDE.md's locked decisions.

Usage::

    uv run python -m volrisk.db.migrate
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Engine, text

from volrisk.db.engine import get_engine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path("db/migrations")


def apply_migrations(engine: Engine, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending ``*.sql`` files in name order; return the versions applied."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
        )
        done = {row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))}

    applied: list[str] = []
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        version = sql_file.stem
        if version in done:
            logger.info("migration %s already applied", version)
            continue
        with engine.begin() as conn:  # one transaction per migration: applied fully or not at all
            conn.execute(text(sql_file.read_text(encoding="utf-8")))
            conn.execute(
                text("INSERT INTO schema_migrations (version) VALUES (:v)"), {"v": version}
            )
        logger.info("applied migration %s", version)
        applied.append(version)
    return applied


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    applied = apply_migrations(get_engine())
    print(f"migrations applied: {applied if applied else 'none (up to date)'}")


if __name__ == "__main__":
    main()
