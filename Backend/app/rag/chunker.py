"""
Document chunking.

Uses `langchain-text-splitters` (a lightweight standalone package, not the
full LangChain framework) for a battle-tested recursive splitter, wrapped
behind our own function so callers depend on our signature, not LangChain's.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(slots=True)
class Chunk:
    text: str
    chunk_index: int


def chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split raw document text into overlapping, semantically-reasonable chunks."""
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [Chunk(text=piece, chunk_index=i) for i, piece in enumerate(pieces) if piece.strip()]
