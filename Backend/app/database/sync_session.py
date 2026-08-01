"""
Synchronous DB session for Celery workers.

FastAPI request handlers use the async engine (`app/database/session.py`);
Celery's worker execution model is synchronous, so tasks get their own
plain `psycopg2`-backed sync engine instead of trying to run an event loop
inside a worker process. Both point at the same database and share the
same `Base`/ORM models — only the driver and concurrency model differ.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def _to_sync_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg2://", 1)
    return dsn


_settings = get_settings()
sync_engine = create_engine(_to_sync_dsn(_settings.DATABASE_URL), echo=_settings.DATABASE_ECHO, future=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False, class_=Session)


@contextmanager
def get_sync_db_session():
    session = SyncSessionLocal()
    try:
        yield session
    finally:
        session.close()
