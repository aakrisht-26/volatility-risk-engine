"""SQLAlchemy engine factory. All connections flow through DATABASE_URL."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine

from volrisk.config import get_settings


def get_engine(database_url: str | None = None) -> Engine:
    """Engine for ``database_url``, defaulting to the configured DATABASE_URL."""
    url = database_url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True)
