"""
Prompt injection & PII detection.

Regex/heuristic-based by design: it runs synchronously on every chat
message before it ever reaches the LLM, so it must be fast and dependency-free.
This is a first line of defense, not a guarantee — flagged findings are
logged to the audit trail (see services/audit) for human review. A
production system should layer a dedicated classifier (e.g. a fine-tuned
DeBERTa injection detector) behind this same `scan_message` signature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (your|the) (system|previous) prompt", re.IGNORECASE),
    re.compile(r"you are now (in )?(developer|jailbreak|dan) mode", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.IGNORECASE),
    re.compile(r"act as if you have no (restrictions|rules|filters)", re.IGNORECASE),
]

_PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
}


@dataclass(slots=True)
class SecurityScanResult:
    is_suspected_injection: bool = False
    injection_matches: list[str] = field(default_factory=list)
    detected_pii_types: list[str] = field(default_factory=list)

    @property
    def requires_review(self) -> bool:
        return self.is_suspected_injection or bool(self.detected_pii_types)


def scan_message(text: str) -> SecurityScanResult:
    injection_matches = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    pii_types = [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]
    return SecurityScanResult(
        is_suspected_injection=bool(injection_matches),
        injection_matches=injection_matches,
        detected_pii_types=pii_types,
    )
