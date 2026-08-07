"""
Mock SAP connector.

Deliberately mirrors real SAP OData/Gateway field naming conventions
(MATNR, MAKTX, LFA1-style vendor fields, EBELN, RE-document numbering)
so that swapping in a real `SAPGatewayConnector` later is a drop-in
replacement with no shape surprises for calling code or the RAG pipeline's
prompt templates.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.sap.base import SAPConnector

_MATERIALS: dict[str, dict[str, Any]] = {
    "MAT-100": {
        "MATNR": "MAT-100",
        "MAKTX": "Steel Rod 10mm",
        "MEINS": "EA",
        "LABST": 4820,
        "PLANT": "1000",
        "MTART": "ROH",
    },
    "MAT-200": {
        "MATNR": "MAT-200",
        "MAKTX": "Industrial Bearing Assembly",
        "MEINS": "EA",
        "LABST": 312,
        "PLANT": "1000",
        "MTART": "ROH",
    },
}

_VENDORS: dict[str, dict[str, Any]] = {
    "VEN-500": {"LIFNR": "VEN-500", "NAME1": "Bosch Industrial Supplies", "LAND1": "DE", "ORT01": "Stuttgart"},
    "VEN-600": {"LIFNR": "VEN-600", "NAME1": "Tata Steel Ltd", "LAND1": "IN", "ORT01": "Jamshedpur"},
}

_PURCHASE_ORDERS: list[dict[str, Any]] = [
    {"EBELN": "4500001234", "LIFNR": "VEN-500", "STATUS": "Open", "NETWR": 18500.00, "WAERS": "EUR", "BEDAT": "2026-06-02"},
    {"EBELN": "4500001235", "LIFNR": "VEN-600", "STATUS": "Open", "NETWR": 9200.00, "WAERS": "INR", "BEDAT": "2026-06-10"},
    {"EBELN": "4500001236", "LIFNR": "VEN-500", "STATUS": "Closed", "NETWR": 4300.00, "WAERS": "EUR", "BEDAT": "2026-05-18"},
]

_INVOICES: dict[str, dict[str, Any]] = {
    "51003": {
        "BELNR": "51003",
        "EBELN": "4500001234",
        "LIFNR": "VEN-500",
        "AMOUNT": 18500.00,
        "WAERS": "EUR",
        "STATUS": "Approved",
        "APPROVED_BY": "R. Fischer (AP Manager)",
        "APPROVAL_DATE": "2026-06-15",
    },
}

_SIMULATED_LATENCY_SECONDS = 0.05


class MockSAPConnector(SAPConnector):
    provider_name = "mock-sap"

    async def get_material(self, material_id: str) -> dict[str, Any] | None:
        await asyncio.sleep(_SIMULATED_LATENCY_SECONDS)
        return _MATERIALS.get(material_id.upper())

    async def get_vendor(self, vendor_id: str) -> dict[str, Any] | None:
        await asyncio.sleep(_SIMULATED_LATENCY_SECONDS)
        return _VENDORS.get(vendor_id.upper())

    async def get_purchase_orders(self, status: str | None = None) -> list[dict[str, Any]]:
        await asyncio.sleep(_SIMULATED_LATENCY_SECONDS)
        if status is None:
            return list(_PURCHASE_ORDERS)
        return [po for po in _PURCHASE_ORDERS if po["STATUS"].lower() == status.lower()]

    async def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        await asyncio.sleep(_SIMULATED_LATENCY_SECONDS)
        return _INVOICES.get(invoice_id)
