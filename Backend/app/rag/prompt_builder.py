"""
Prompt construction.

Deliberately isolated from the LLM call itself so prompt engineering can be
iterated on (and unit tested) without touching provider code. Each context
chunk is tagged with a stable `[S1]`, `[S2]`... marker; the LLM is instructed
to cite by marker, and `citation.py` resolves those markers back to real
source documents.
"""
from __future__ import annotations

from app.llm.base import ChatMessage, ChatRole
from app.rag.hybrid_retriever import RetrievedChunk

_SYSTEM_PROMPT = """You are the AI SAP Knowledge Assistant, an enterprise assistant for SAP \
business questions. Answer ONLY using the numbered source excerpts provided in the user \
message. Every factual claim must end with the marker(s) of the source(s) it came from, \
like [S1] or [S1][S3].

Rules:
- If the sources do not contain the answer, say so explicitly. Do not invent SAP data.
- Prefer precise, actionable answers (transaction codes, field names, exact figures) over \
vague summaries.
- Keep the tone professional and concise, as an SAP functional consultant would.
"""


def build_messages(query: str, context_chunks: list[RetrievedChunk], conversation_history: list[ChatMessage]) -> list[ChatMessage]:
    source_block_lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_label = chunk.metadata.get("source_name", chunk.metadata.get("source_type", "document"))
        source_block_lines.append(f"[S{i}] (source: {source_label})\n{chunk.document}")
    source_block = "\n\n".join(source_block_lines) if source_block_lines else "No relevant sources were found."

    user_content = f"Sources:\n{source_block}\n\nQuestion: {query}"

    messages = [ChatMessage(role=ChatRole.SYSTEM, content=_SYSTEM_PROMPT)]
    messages.extend(conversation_history)
    messages.append(ChatMessage(role=ChatRole.USER, content=user_content))
    return messages
