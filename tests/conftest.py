"""Shared test fixtures. DB fixtures touch only the disposable test database."""

import os

import pytest

TEST_DB_URL = os.environ.get("VOLRISK_TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def db_engine():
    if not TEST_DB_URL:
        pytest.skip("VOLRISK_TEST_DATABASE_URL not set")
    from volrisk.db.engine import get_engine
    from volrisk.db.migrate import apply_migrations

    engine = get_engine(TEST_DB_URL)
    apply_migrations(engine)
    yield engine
    engine.dispose()
