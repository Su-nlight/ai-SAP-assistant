"""Resolves the [S1][S2]-style markers the LLM was instructed to emit back into real source metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.hybrid_retriever import RetrievedChunk

_MARKER_PATTERN = re.compile(r"\[S(\d+)\]")


@dataclass(slots=True)
class Citation:
    marker: str
    chunk_id: str
    source_name: str
    source_type: str
    excerpt: str


def extract_citations(answer_text: str, context_chunks: list[RetrievedChunk]) -> list[Citation]:
    used_indices = {int(m) for m in _MARKER_PATTERN.findall(answer_text)}
    citations: list[Citation] = []
    for i, chunk in enumerate(context_chunks, start=1):
        if i not in used_indices:
            continue
        citations.append(
            Citation(
                marker=f"[S{i}]",
                chunk_id=chunk.id,
                source_name=chunk.metadata.get("source_name", "unknown"),
                source_type=chunk.metadata.get("source_type", "document"),
                excerpt=chunk.document[:280],
            )
        )
    return citations
