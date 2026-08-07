"""SAP connector factory. Real SAP Gateway/OData connector plugs in here behind the same interface."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.sap.base import SAPConnector
from app.sap.mock_sap import MockSAPConnector


def build_sap_connector(settings: Settings) -> SAPConnector:
    match settings.SAP_MODE:
        case "mock":
            return MockSAPConnector()
        case "real":
            from app.sap.sap_gateway_connector import SAPGatewayConnector

            return SAPGatewayConnector(
                base_url=settings.SAP_GATEWAY_BASE_URL or "",
                token_url=settings.SAP_TOKEN_URL or "",
                client_id=settings.SAP_CLIENT_ID or "",
                client_secret=settings.SAP_CLIENT_SECRET or "",
                material_entity_set=settings.SAP_ODATA_MATERIAL_ENTITY_SET,
                vendor_entity_set=settings.SAP_ODATA_VENDOR_ENTITY_SET,
                po_entity_set=settings.SAP_ODATA_PO_ENTITY_SET,
                invoice_entity_set=settings.SAP_ODATA_INVOICE_ENTITY_SET,
                timeout_seconds=settings.SAP_HTTP_TIMEOUT_SECONDS,
            )
            # RFC/BAPI/IDoc connectors are future increments (see docs/ARCHITECTURE.md) —
            # they'd need PyRFC (requires the proprietary SAP NW RFC SDK) rather
            # than plain HTTP, so they're kept as a separate connector class
            # implementing this same `SAPConnector` interface.
        case other:
            raise NotImplementedError(f"Unknown SAP_MODE '{other}'.")


@lru_cache
def get_sap_connector() -> SAPConnector:
    return build_sap_connector(get_settings())
