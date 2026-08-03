"""
Query rewriting + metadata extraction.

Rule-based on purpose: these run on every single query, so keeping them
free and sub-millisecond (instead of an extra LLM round-trip per query)
matters far more than squeezing out marginal accuracy. SAP artifacts follow
predictable, high-signal patterns (transaction codes are 3-6 uppercase
alphanumerics; material/PO/invoice numbers follow known prefixes), so
regex extraction is a good fit here.

If this stage's precision ever becomes the bottleneck, promote it to an
LLM-based extractor behind the same function signature — no caller changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TCODE_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{1,4})\b")
_MATERIAL_PATTERN = re.compile(r"\bMAT-[A-Z0-9]+\b", re.IGNORECASE)
_PO_PATTERN = re.compile(r"\bPO-[A-Z0-9]+\b|\b\d{10}\b", re.IGNORECASE)
_INVOICE_PATTERN = re.compile(r"\bINV-[A-Z0-9]+\b|\bInvoice\s+(\d+)\b", re.IGNORECASE)

_STOPWORDS_FOR_REWRITE = {"please", "could", "you", "kindly", "can", "tell", "me"}


@dataclass(slots=True)
class QueryMetadata:
    transaction_codes: list[str] = field(default_factory=list)
    material_ids: list[str] = field(default_factory=list)
    po_numbers: list[str] = field(default_factory=list)
    invoice_numbers: list[str] = field(default_factory=list)

    @property
    def has_sap_entity(self) -> bool:
        return any([self.transaction_codes, self.material_ids, self.po_numbers, self.invoice_numbers])


def extract_query_metadata(query: str) -> QueryMetadata:
    tcodes = [m for m in _TCODE_PATTERN.findall(query) if m.isupper() and len(m) >= 4]
    materials = _MATERIAL_PATTERN.findall(query)
    pos = _PO_PATTERN.findall(query)
    invoice_direct = _INVOICE_PATTERN.findall(query)
    invoices = [m if isinstance(m, str) and m else m for m in invoice_direct]
    return QueryMetadata(
        transaction_codes=list(dict.fromkeys(tcodes)),
        material_ids=list(dict.fromkeys(materials)),
        po_numbers=list(dict.fromkeys(pos)),
        invoice_numbers=[m for m in dict.fromkeys(invoices) if m],
    )


def rewrite_query(query: str) -> str:
    """Strip conversational filler so retrieval keys in on content words."""
    words = query.strip().split()
    filtered = [w for w in words if w.lower().strip("?,.") not in _STOPWORDS_FOR_REWRITE]
    rewritten = " ".join(filtered) if filtered else query
    return rewritten
