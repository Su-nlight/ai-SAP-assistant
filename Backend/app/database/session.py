"""SQLAlchemy engine/session wiring. Async engine so FastAPI routes never block on DB I/O."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in the project."""


def _to_async_dsn(dsn: str) -> str:
    """Accept a plain postgresql:// DSN from .env and adapt it for asyncpg."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    if dsn.startswith("postgresql+psycopg2://"):
        return dsn.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    return dsn


_settings = get_settings()
engine = create_async_engine(_to_async_dsn(_settings.DATABASE_URL), echo=_settings.DATABASE_ECHO, future=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — one session per request, always closed afterward."""
    async with AsyncSessionLocal() as session:
        yield session
