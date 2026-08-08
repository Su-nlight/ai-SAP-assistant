"""
Dependency injection wiring for FastAPI routes.

This module is intentionally the *only* place that knows how to assemble
a `ChatService`, `DocumentService`, etc. from their dependencies. Routes
depend on these factory functions, never on concrete repository/service
classes directly — this is what makes the layering enforceable rather
than aspirational.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import TokenError, decode_token
from app.database.session import get_db_session
from app.embeddings.factory import get_embedding_provider
from app.llm.factory import get_llm_provider
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import CrossEncoderReranker
from app.repositories.audit_repository import AuditLogRepository
from app.repositories.chat_repository import ChatRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.user_repository import RoleRepository, UserRepository
from app.sap.factory import get_sap_connector
from app.services.audit_service import AuditService
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.sap_service import SAPService
from app.vectorstore.factory import get_vector_store

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


@dataclass(slots=True)
class CurrentUser:
    id: uuid.UUID
    role: str


async def get_current_user(
    token: str | None = Depends(_oauth2_scheme),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    try:
        payload = decode_token(token, settings.JWT_SECRET_KEY, settings.JWT_ALGORITHM)
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return CurrentUser(id=uuid.UUID(payload["sub"]), role=payload["role"])


# --------------------------------------------------------------------- repositories

def get_user_repository(session: AsyncSession = Depends(get_db_session)) -> UserRepository:
    return UserRepository(session)


def get_role_repository(session: AsyncSession = Depends(get_db_session)) -> RoleRepository:
    return RoleRepository(session)


def get_document_repository(session: AsyncSession = Depends(get_db_session)) -> DocumentRepository:
    return DocumentRepository(session)


def get_chat_repository(session: AsyncSession = Depends(get_db_session)) -> ChatRepository:
    return ChatRepository(session)


def get_audit_log_repository(session: AsyncSession = Depends(get_db_session)) -> AuditLogRepository:
    return AuditLogRepository(session)


def get_audit_service(audit_repo: AuditLogRepository = Depends(get_audit_log_repository)) -> AuditService:
    return AuditService(audit_repo)


# --------------------------------------------------------------------- RAG pipeline

def get_rag_pipeline(settings: Settings = Depends(get_settings)) -> RAGPipeline:
    hybrid_retriever = HybridRetriever(
        vector_store=get_vector_store(),
        embedder=get_embedding_provider(),
        dense_weight=settings.HYBRID_DENSE_WEIGHT,
        sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
    )
    return RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=CrossEncoderReranker(),
        llm_provider=get_llm_provider(),
        retrieval_top_k=settings.RETRIEVAL_TOP_K,
        rerank_top_k=settings.RERANK_TOP_K,
        max_context_tokens=settings.LLM_MAX_TOKENS * 2,
    )


# --------------------------------------------------------------------- services

def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
    role_repo: RoleRepository = Depends(get_role_repository),
    settings: Settings = Depends(get_settings),
) -> AuthService:
    return AuthService(user_repo, role_repo, settings)


def get_document_service(
    document_repo: DocumentRepository = Depends(get_document_repository),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    return DocumentService(
        document_repo=document_repo,
        vector_store=get_vector_store(),
        embedder=get_embedding_provider(),
        settings=settings,
    )


def get_chat_service(
    chat_repo: ChatRepository = Depends(get_chat_repository),
    document_repo: DocumentRepository = Depends(get_document_repository),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    audit_service: AuditService = Depends(get_audit_service),
) -> ChatService:
    return ChatService(
        chat_repo=chat_repo,
        document_repo=document_repo,
        rag_pipeline=rag_pipeline,
        sap_connector=get_sap_connector(),
        llm_provider=get_llm_provider(),
        audit_service=audit_service,
    )


def get_sap_service(session: AsyncSession = Depends(get_db_session)) -> SAPService:
    return SAPService(connector=get_sap_connector(), session=session)
