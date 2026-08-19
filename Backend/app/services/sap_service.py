"""
Application-layer SAP access, with a write-through cache into `sap_cache`.

Real SAP Gateway calls are comparatively slow and rate-limited; caching
recent lookups (materials, vendors, POs, invoices) reduces load on SAP and
gives the RAG pipeline near-instant repeated lookups within a conversation.
Cache invalidation policy (TTL, explicit bust) is a documented next
increment — this establishes the write-through mechanics correctly today.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.audit_feedback import SAPCache
from app.sap.base import SAPConnector


class SAPService:
    def __init__(self, connector: SAPConnector, session: AsyncSession) -> None:
        self._connector = connector
        self._session = session

    async def get_material(self, material_id: str) -> dict:
        data = await self._connector.get_material(material_id)
        if data is None:
            raise NotFoundError(f"Material '{material_id}' not found.")
        await self._write_cache("material", material_id, data)
        return data

    async def get_vendor(self, vendor_id: str) -> dict:
        data = await self._connector.get_vendor(vendor_id)
        if data is None:
            raise NotFoundError(f"Vendor '{vendor_id}' not found.")
        await self._write_cache("vendor", vendor_id, data)
        return data

    async def get_purchase_orders(self, status: str | None) -> list[dict]:
        return await self._connector.get_purchase_orders(status)

    async def get_invoice(self, invoice_id: str) -> dict:
        data = await self._connector.get_invoice(invoice_id)
        if data is None:
            raise NotFoundError(f"Invoice '{invoice_id}' not found.")
        await self._write_cache("invoice", invoice_id, data)
        return data

    async def _write_cache(self, object_type: str, object_id: str, data: dict) -> None:
        existing = await self._session.execute(
            select(SAPCache).where(SAPCache.object_type == object_type, SAPCache.object_id == object_id)
        )
        row = existing.scalar_one_or_none()
        if row is None:
            self._session.add(SAPCache(object_type=object_type, object_id=object_id, json_data=data))
        else:
            row.json_data = data
        await self._session.commit()
