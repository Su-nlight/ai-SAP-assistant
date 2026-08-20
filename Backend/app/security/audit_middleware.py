"""
Generic audit middleware.

Runs before routing/dependency-injection, so it can't use `Depends(...)` —
it opens its own short-lived DB session instead. Deliberately best-effort:
a failure to *write* an audit record must never fail the underlying
request, so all DB errors here are caught and logged, not raised.
"""
from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import Settings
from app.core.security import TokenError, decode_token
from app.database.session import AsyncSessionLocal
from app.repositories.audit_repository import AuditLogRepository
from app.services.audit_service import AuditService

logger = logging.getLogger("ai_sap_assistant.audit")

_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# Auth endpoints are logged with special-cased actions (no user_id resolvable
# for a still-anonymous request) rather than skipped outright — a failed
# login attempt is exactly the kind of event an audit trail exists for.
_EXCLUDED_PATH_PREFIXES = ("/docs", "/openapi.json", "/redoc")


class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        if not self._settings.AUDIT_LOG_MUTATING_REQUESTS:
            return response
        if request.method not in _MUTATING_METHODS:
            return response
        if any(request.url.path.startswith(p) for p in _EXCLUDED_PATH_PREFIXES):
            return response

        try:
            await self._write_audit_entry(request, response)
        except Exception:  # noqa: BLE001 — audit logging must never break the request
            logger.exception("Failed to write audit log entry for %s %s", request.method, request.url.path)

        return response

    async def _write_audit_entry(self, request: Request, response: Response) -> None:
        user_id = self._extract_user_id(request)
        client_ip = request.client.host if request.client else None

        async with AsyncSessionLocal() as session:
            audit_service = AuditService(AuditLogRepository(session))
            await audit_service.record(
                action=f"{request.method} {request.url.path}",
                user_id=user_id,
                resource=request.url.path,
                ip_address=client_ip,
                details={"status_code": response.status_code},
            )

    def _extract_user_id(self, request: Request):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None
        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_token(token, self._settings.JWT_SECRET_KEY, self._settings.JWT_ALGORITHM)
        except TokenError:
            return None
        import uuid

        try:
            return uuid.UUID(payload["sub"])
        except (KeyError, ValueError):
            return None
