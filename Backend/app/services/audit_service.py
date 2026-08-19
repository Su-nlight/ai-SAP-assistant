"""
Audit logging.

Two call sites feed this service:
  1. `AuditMiddleware` (app/security/audit_middleware.py) — generic coverage
     of every authenticated mutating request (POST/PUT/DELETE), logged with
     no knowledge of what the route does.
  2. Targeted calls from services that have context the middleware can't
     see (e.g. `ChatService` logging *why* a message was flagged — a
     suspected prompt injection payload, not just "POST /api/chat happened").

Both paths converge on this one service so there is a single, consistent
shape for every audit row and a single place to change if the audit
schema ever evolves.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_feedback import AuditLog
from app.repositories.audit_repository import AuditLogRepository


class AuditService:
    def __init__(self, repo: AuditLogRepository) -> None:
        self._repo = repo

    async def record(
        self,
        action: str,
        user_id: uuid.UUID | None = None,
        resource: str | None = None,
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            details_json=details or {},
        )
        entry = await self._repo.add(entry)
        await self._repo.commit()
        return entry

    async def list_recent(self, limit: int = 100) -> list[AuditLog]:
        return await self._repo.list_recent(limit=limit)


def get_audit_service(session: AsyncSession) -> AuditService:
    return AuditService(AuditLogRepository(session))
