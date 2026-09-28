"""
src/api/deps.py
──────────────────
FastAPI dependencies. get_db is created lazily (not at import time) so
importing this module never requires a reachable database -- readiness
checks and non-DB endpoints must keep working even if Postgres is down.
"""

from typing import Generator

from sqlalchemy.orm import Session

from src.db.session import get_engine, get_session_factory

_session_factory = None


def _factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory(get_engine())
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    db = _factory()()
    try:
        yield db
    finally:
        db.close()
