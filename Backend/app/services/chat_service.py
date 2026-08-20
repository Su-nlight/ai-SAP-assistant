"""Application-layer chat orchestration: security scan -> SAP context -> RAG -> persistence."""
from __future__ import annotations

import uuid

from app.core.metrics import rag_queries_total
from app.llm.base import ChatMessage, ChatRole, LLMProvider
from app.models.audit_feedback import Feedback
from app.models.chat import ChatSession, Message
from app.rag.bm25_retriever import BM25Document
from app.rag.pipeline import RAGPipeline
from app.rag.query_processing import extract_query_metadata
from app.repositories.chat_repository import ChatRepository
from app.repositories.document_repository import DocumentRepository
from app.sap.base import SAPConnector
from app.security.prompt_injection import scan_message
from app.services.audit_service import AuditService
from app.services.sap_context_adapter import fetch_open_purchase_orders_if_relevant, fetch_sap_context


class ChatService:
    def __init__(
        self,
        chat_repo: ChatRepository,
        document_repo: DocumentRepository,
        rag_pipeline: RAGPipeline,
        sap_connector: SAPConnector,
        llm_provider: LLMProvider,
        audit_service: AuditService | None = None,
    ) -> None:
        self._chat_repo = chat_repo
        self._document_repo = document_repo
        self._pipeline = rag_pipeline
        self._sap = sap_connector
        self._llm = llm_provider
        self._audit = audit_service

    async def send_message(self, user_id: uuid.UUID, session_id: uuid.UUID | None, message_text: str) -> Message:
        security_result = scan_message(message_text)
        if security_result.requires_review and self._audit is not None:
            # The system prompt already constrains the model to grounded,
            # citation-backed answers, which neutralizes most injection
            # payloads — so this is not auto-blocked, only flagged for
            # human/admin visibility via the audit trail.
            await self._audit.record(
                action="chat.security_flag",
                user_id=user_id,
                resource="chat_message",
                details={
                    "is_suspected_injection": security_result.is_suspected_injection,
                    "detected_pii_types": security_result.detected_pii_types,
                },
            )

        session = await self._get_or_create_session(user_id, session_id)

        history_rows = await self._chat_repo.get_recent_messages(session.id, limit=10)
        history = self._rows_to_chat_messages(history_rows)

        chunks = await self._document_repo.get_all_chunks()
        bm25_corpus = [
            BM25Document(id=c.embedding_id, text=c.chunk_text, metadata={"document_id": str(c.document_id)})
            for c in chunks
        ]

        query_metadata = extract_query_metadata(message_text)
        pinned = await fetch_sap_context(self._sap, query_metadata)
        pinned += await fetch_open_purchase_orders_if_relevant(self._sap, message_text)

        rag_response = await self._pipeline.run(
            query=message_text,
            bm25_corpus=bm25_corpus,
            conversation_history=history,
            pinned_chunks=pinned or None,
        )
        rag_queries_total.labels(llm_provider=rag_response.provider, outcome="success").inc()

        message = Message(
            session_id=session.id,
            role="assistant",
            question=message_text,
            answer=rag_response.answer,
            citations_json=[
                {
                    "marker": c.marker,
                    "source_name": c.source_name,
                    "source_type": c.source_type,
                    "excerpt": c.excerpt,
                }
                for c in rag_response.citations
            ],
            provider=rag_response.provider,
            model=rag_response.model,
            latency_ms=int(rag_response.latency_ms),
            total_tokens=rag_response.total_tokens,
        )
        message = await self._chat_repo.add_message(message)

        if session.title == "New Conversation":
            session.title = await self._generate_title(message_text)

        await self._chat_repo.commit()
        return message

    async def record_feedback(self, user_id: uuid.UUID, message_id: uuid.UUID, is_positive: bool, comment: str | None) -> Feedback:
        feedback = Feedback(
            message_id=message_id,
            user_id=user_id,
            is_positive=is_positive,
            comment=comment,
            flagged_for_eval=not is_positive,  # feedback loop: 👎 responses become eval candidates
        )
        feedback = await self._chat_repo.add_feedback(feedback)
        await self._chat_repo.commit()
        return feedback

    async def _get_or_create_session(self, user_id: uuid.UUID, session_id: uuid.UUID | None) -> ChatSession:
        if session_id is not None:
            existing = await self._chat_repo.get_session_with_messages(session_id, user_id)
            if existing is not None:
                return existing
        new_session = ChatSession(user_id=user_id, title="New Conversation")
        return await self._chat_repo.add(new_session)

    async def _generate_title(self, first_message: str) -> str:
        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="Generate a concise 4-6 word title for a chat conversation. Reply with only the title.",
            ),
            ChatMessage(role=ChatRole.USER, content=first_message),
        ]
        try:
            response = await self._llm.generate(messages, temperature=0.3, max_tokens=20)
            return response.content.strip().strip('"')[:100] or "New Conversation"
        except Exception:  # noqa: BLE001 — title generation must never break the chat flow
            return first_message[:60]

    @staticmethod
    def _rows_to_chat_messages(rows: list[Message]) -> list[ChatMessage]:
        history: list[ChatMessage] = []
        for row in rows:
            if row.question:
                history.append(ChatMessage(role=ChatRole.USER, content=row.question))
            if row.answer:
                history.append(ChatMessage(role=ChatRole.ASSISTANT, content=row.answer))
        return history
