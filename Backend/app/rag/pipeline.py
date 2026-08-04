"""
RAG pipeline orchestration.

Implements the exact stage sequence from the project spec:

    Query -> Query Rewriting -> Metadata Extraction -> Retriever ->
    Hybrid Retrieval -> Reranker -> Context Compression -> Prompt Builder ->
    LLM -> Citation Generator -> Response

This class has no FastAPI, database, or HTTP knowledge whatsoever — it is
pure application-layer logic, callable from an API route, a Celery worker,
or a test, identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from app.llm.base import ChatMessage, LLMProvider
from app.rag.bm25_retriever import BM25Document
from app.rag.citation import Citation, extract_citations
from app.rag.context_compressor import compress_context
from app.rag.hybrid_retriever import HybridRetriever, RetrievedChunk
from app.rag.prompt_builder import build_messages
from app.rag.query_processing import QueryMetadata, extract_query_metadata, rewrite_query
from app.rag.reranker import CrossEncoderReranker


@dataclass(slots=True)
class RAGResponse:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    query_metadata: QueryMetadata
    model: str
    provider: str
    latency_ms: float
    total_tokens: int | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_provider: LLMProvider,
        retrieval_top_k: int,
        rerank_top_k: int,
        max_context_tokens: int,
    ) -> None:
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._llm = llm_provider
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._max_context_tokens = max_context_tokens

    async def run(
        self,
        query: str,
        bm25_corpus: list[BM25Document],
        conversation_history: list[ChatMessage] | None = None,
        metadata_filter: dict | None = None,
        pinned_chunks: list[RetrievedChunk] | None = None,
    ) -> RAGResponse:
        """
        `pinned_chunks` lets a caller (e.g. ChatService, after querying live
        SAP data) guarantee specific context makes it into the prompt
        regardless of how the reranker scores it — live transactional data
        for a matched material/PO/invoice number is authoritative and
        should not be able to lose a relevance contest against document
        chunks.
        """
        timings: dict[str, float] = {}

        t0 = perf_counter()
        rewritten_query = rewrite_query(query)
        timings["query_rewriting"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        query_metadata = extract_query_metadata(query)
        timings["metadata_extraction"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        candidates = self._retriever.retrieve(
            rewritten_query, top_k=self._retrieval_top_k, bm25_corpus=bm25_corpus, metadata_filter=metadata_filter
        )
        timings["hybrid_retrieval"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        reranked = self._reranker.rerank(rewritten_query, candidates, top_k=self._rerank_top_k)
        timings["reranking"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        combined = (pinned_chunks or []) + reranked
        compressed = compress_context(combined, max_context_tokens=self._max_context_tokens)
        timings["context_compression"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        messages = build_messages(query, compressed, conversation_history or [])
        timings["prompt_building"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        llm_response = await self._llm.generate(messages)
        timings["llm_generation"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        citations = extract_citations(llm_response.content, compressed)
        timings["citation_generation"] = (perf_counter() - t0) * 1000

        total_latency = sum(timings.values())

        return RAGResponse(
            answer=llm_response.content,
            citations=citations,
            retrieved_chunks=compressed,
            query_metadata=query_metadata,
            model=llm_response.model,
            provider=llm_response.provider,
            latency_ms=total_latency,
            total_tokens=llm_response.total_tokens,
            stage_timings_ms=timings,
        )
