"""
src/db/session.py
───────────────────
Engine/session setup. Reads DATABASE_URL from the environment (or a local
.env, if python-dotenv and one are present); falls back to a plain local
Postgres default so `python sync_run_to_db.py` works out of the box after
`createdb credit_line_manager`. See .env.example.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg2://localhost:5432/credit_line_manager"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(url: str | None = None):
    return create_engine(url or DATABASE_URL, future=True)


def get_session_factory(engine=None) -> sessionmaker:
    return sessionmaker(bind=engine or get_engine(), future=True, expire_on_commit=False)


def get_session(engine=None) -> Session:
    """One-off session, e.g. for a script. Caller is responsible for
    closing it (or use as a context manager)."""
    return get_session_factory(engine)()
