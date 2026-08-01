"""
Cross-dialect column types.

Production runs on PostgreSQL and should get native `UUID` and `JSONB`
(indexable, efficient). The test suite runs against in-memory SQLite for
speed/isolation and needs portable equivalents. SQLAlchemy's `with_variant`
gives us both from one declaration — no per-dialect branching anywhere else
in the codebase.
"""
from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy import Uuid as PortableUUID
from sqlalchemy.dialects.postgresql import JSONB

PortableJSON = JSON().with_variant(JSONB, "postgresql")

__all__ = ["PortableUUID", "PortableJSON"]
