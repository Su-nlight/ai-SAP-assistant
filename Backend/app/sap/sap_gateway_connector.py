"""
Real SAP Gateway connector.

Talks to actual SAP Gateway OData v2 services over OAuth2 client-credentials
— the standard pattern for a headless service account calling SAP Gateway
(SAP BTP-fronted or on-prem via SAP Cloud Connector). Same `SAPConnector`
interface as `MockSAPConnector`, so the RAG pipeline, `SAPService`, and every
route above them work identically against mock or real SAP with a single
`SAP_MODE=real` config flip.

OData v2 wraps every payload in a `{"d": {...}}` envelope (v4 does not) —
this connector targets v2 since it remains the dominant version across
existing SAP Gateway/Fiori deployments. Swapping to v4 would only touch
`_unwrap_odata_response` and the query-string builder, not this class's
public interface.
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.sap.base import SAPConnector


class SAPAuthError(RuntimeError):
    """Raised when OAuth2 client-credentials token acquisition fails."""


class SAPGatewayConnector(SAPConnector):
    provider_name = "sap-gateway"

    def __init__(
        self,
        base_url: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        material_entity_set: str,
        vendor_entity_set: str,
        po_entity_set: str,
        invoice_entity_set: str,
        timeout_seconds: int = 30,
    ) -> None:
        if not (base_url and token_url and client_id and client_secret):
            raise SAPAuthError(
                "SAP_GATEWAY_BASE_URL, SAP_TOKEN_URL, SAP_CLIENT_ID and SAP_CLIENT_SECRET are all required for SAP_MODE=real."
            )
        self._base_url = base_url.rstrip("/")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._entity_sets = {
            "material": material_entity_set,
            "vendor": vendor_entity_set,
            "po": po_entity_set,
            "invoice": invoice_entity_set,
        }
        self._timeout = timeout_seconds
        self._cached_token: str | None = None
        self._token_expires_at: float = 0.0

    async def _get_access_token(self) -> str:
        # Refresh 30s before actual expiry to avoid a request racing an
        # about-to-expire token.
        if self._cached_token and time.monotonic() < self._token_expires_at - 30:
            return self._cached_token

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if response.status_code != 200:
            raise SAPAuthError(f"SAP OAuth2 token request failed ({response.status_code}): {response.text}")

        payload = response.json()
        self._cached_token = payload["access_token"]
        self._token_expires_at = time.monotonic() + int(payload.get("expires_in", 3600))
        return self._cached_token

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    async def _odata_get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
        token = await self._get_access_token()
        url = f"{self._base_url}/{path}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                raise httpx.TransportError(f"SAP Gateway unreachable at {url}: {exc}") from exc

        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"SAP Gateway error {response.status_code}: {response.text}", request=response.request, response=response
            )
        return response.json()

    @staticmethod
    def _unwrap_odata_response(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """OData v2 wraps a single-entity GET as `{"d": {...fields...}}`."""
        if payload is None:
            return None
        return payload.get("d", payload)

    @staticmethod
    def _unwrap_odata_collection(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        """OData v2 wraps a collection GET as `{"d": {"results": [...]}}`."""
        if payload is None:
            return []
        inner = payload.get("d", payload)
        return inner.get("results", [])

    async def get_material(self, material_id: str) -> dict[str, Any] | None:
        entity_set = self._entity_sets["material"]
        payload = await self._odata_get(f"{entity_set}('{material_id}')")
        return self._unwrap_odata_response(payload)

    async def get_vendor(self, vendor_id: str) -> dict[str, Any] | None:
        entity_set = self._entity_sets["vendor"]
        payload = await self._odata_get(f"{entity_set}('{vendor_id}')")
        return self._unwrap_odata_response(payload)

    async def get_purchase_orders(self, status: str | None = None) -> list[dict[str, Any]]:
        entity_set = self._entity_sets["po"]
        params = {"$format": "json"}
        if status:
            params["$filter"] = f"Status eq '{status}'"
        payload = await self._odata_get(entity_set, params=params)
        return self._unwrap_odata_collection(payload)

    async def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        entity_set = self._entity_sets["invoice"]
        payload = await self._odata_get(f"{entity_set}('{invoice_id}')")
        return self._unwrap_odata_response(payload)
