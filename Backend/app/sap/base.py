"""SAP connector abstraction (Adapter pattern) — Mock today, real SAP Gateway/OData later, same interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SAPConnector(ABC):
    provider_name: str

    @abstractmethod
    async def get_material(self, material_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_vendor(self, vendor_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_purchase_orders(self, status: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        raise NotImplementedError
