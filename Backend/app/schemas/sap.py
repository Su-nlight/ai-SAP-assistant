from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MaterialResponse(BaseModel):
    MATNR: str
    MAKTX: str
    MEINS: str
    LABST: float
    PLANT: str
    MTART: str


class VendorResponse(BaseModel):
    LIFNR: str
    NAME1: str
    LAND1: str
    ORT01: str


class PurchaseOrderResponse(BaseModel):
    EBELN: str
    LIFNR: str
    STATUS: str
    NETWR: float
    WAERS: str
    BEDAT: str


class InvoiceResponse(BaseModel):
    BELNR: str
    EBELN: str
    LIFNR: str
    AMOUNT: float
    WAERS: str
    STATUS: str
    APPROVED_BY: str | None = None
    APPROVAL_DATE: str | None = None


class SAPErrorResponse(BaseModel):
    detail: str
    context: dict[str, Any] | None = None
