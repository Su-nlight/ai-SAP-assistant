"""
Context compression.

Keeps the prompt within a sane token budget and removes near-duplicate
chunks (common when a document repeats boilerplate across sections) before
they reach the LLM. A cheap character-length heuristic stands in for a
real tokenizer count here — accurate enough for budgeting purposes and
avoids pulling in a model-specific tokenizer for every provider.
"""
from __future__ import annotations

from app.rag.hybrid_retriever import RetrievedChunk

_CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def compress_context(chunks: list[RetrievedChunk], max_context_tokens: int) -> list[RetrievedChunk]:
    seen_normalized: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        normalized = " ".join(chunk.document.lower().split())[:200]
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        deduped.append(chunk)

    kept: list[RetrievedChunk] = []
    budget = max_context_tokens
    for chunk in deduped:
        cost = estimate_tokens(chunk.document)
        if cost > budget and kept:
            break
        kept.append(chunk)
        budget -= cost
    return kept
