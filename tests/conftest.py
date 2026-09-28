"""
Shared fixtures. `db_available` lets DB-dependent tests skip gracefully on
a machine/CI without a reachable Postgres, rather than hard-failing --
consistent with how test_db_models.py/test_sync_run_to_db.py already avoid
requiring a live database (they use SQLite / pure functions). The
Postgres DDL and end-to-end flow have been verified by hand against a real
local instance; see CLAUDE.md's Database section.
"""

import pytest


@pytest.fixture(scope="session")
def api_client():
    """Shared TestClient so the app's lifespan (loads real models/*.pkl +
    rebuilds the ~30k-row feature matrix, a couple seconds) only runs once
    for the whole suite, not once per test."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        from src.db.session import get_engine
        with get_engine().connect():
            return True
    except Exception:
        return False


def require_db(db_available):
    if not db_available:
        pytest.skip("No reachable Postgres for this test (see tests/conftest.py::db_available)")
