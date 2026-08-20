"""
Bridges the SAP connector (structured JSON) and the RAG pipeline (text
chunks with metadata). Keeping this translation in its own module means
neither `sap/` nor `rag/` needs to know the other exists.
"""
from __future__ import annotations

from app.rag.hybrid_retriever import RetrievedChunk
from app.rag.query_processing import QueryMetadata
from app.sap.base import SAPConnector

_PINNED_SCORE = 1.0  # sentinel: always above any organic retrieval/rerank score


def _material_to_text(data: dict) -> str:
    return (
        f"Material {data['MATNR']} ({data['MAKTX']}): current stock is {data['LABST']} {data['MEINS']} "
        f"at plant {data['PLANT']}, material type {data['MTART']}."
    )


def _vendor_to_text(data: dict) -> str:
    return f"Vendor {data['LIFNR']} is {data['NAME1']}, located in {data['ORT01']}, {data['LAND1']}."


def _po_to_text(data: dict) -> str:
    return (
        f"Purchase Order {data['EBELN']} with vendor {data['LIFNR']} is {data['STATUS']}, "
        f"net value {data['NETWR']} {data['WAERS']}, dated {data['BEDAT']}."
    )


def _invoice_to_text(data: dict) -> str:
    base = (
        f"Invoice {data['BELNR']} for PO {data['EBELN']} with vendor {data['LIFNR']}: "
        f"{data['AMOUNT']} {data['WAERS']}, status {data['STATUS']}."
    )
    if data.get("APPROVED_BY"):
        base += f" Approved by {data['APPROVED_BY']} on {data.get('APPROVAL_DATE', 'unknown date')}."
    return base


async def fetch_sap_context(connector: SAPConnector, query_metadata: QueryMetadata) -> list[RetrievedChunk]:
    """Looks up any SAP entities mentioned in the query and returns them as
    pinned, citable context chunks. Returns an empty list when the query
    mentions no recognizable SAP identifiers — the RAG pipeline then relies
    purely on document retrieval, exactly as it should."""
    chunks: list[RetrievedChunk] = []

    for material_id in query_metadata.material_ids:
        data = await connector.get_material(material_id)
        if data:
            chunks.append(_build_chunk(material_id, "sap-material", _material_to_text(data)))

    for po_number in query_metadata.po_numbers:
        pos = await connector.get_purchase_orders()
        match = next((po for po in pos if po["EBELN"] == po_number), None)
        if match:
            chunks.append(_build_chunk(po_number, "sap-purchase-order", _po_to_text(match)))

    for invoice_number in query_metadata.invoice_numbers:
        data = await connector.get_invoice(invoice_number)
        if data:
            chunks.append(_build_chunk(invoice_number, "sap-invoice", _invoice_to_text(data)))

    # "show all open purchase orders" style queries carry no specific id but
    # do carry the word "purchase order" — handled by the caller passing the
    # raw query text through a light keyword check rather than bloating
    # QueryMetadata with free-text intent detection.
    return chunks


async def fetch_open_purchase_orders_if_relevant(connector: SAPConnector, raw_query: str) -> list[RetrievedChunk]:
    lowered = raw_query.lower()
    if "purchase order" not in lowered and " po " not in f" {lowered} ":
        return []
    status = "open" if "open" in lowered else None
    pos = await connector.get_purchase_orders(status=status)
    return [_build_chunk(po["EBELN"], "sap-purchase-order", _po_to_text(po)) for po in pos]


def _build_chunk(entity_id: str, source_type: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"sap:{source_type}:{entity_id}",
        document=text,
        metadata={"source_name": f"SAP live data ({entity_id})", "source_type": source_type},
        dense_score=_PINNED_SCORE,
        sparse_score=_PINNED_SCORE,
        fused_score=_PINNED_SCORE,
    )
