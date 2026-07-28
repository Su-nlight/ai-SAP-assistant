"""
Centralized application configuration.

All environment-driven values live here and nowhere else (no hardcoded
values, per project coding standards). Every subsystem that needs a
setting depends on `Settings` via dependency injection, never on `os.environ`
directly.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- app
    APP_NAME: str = "AI SAP Knowledge Assistant"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api"

    # ---------------------------------------------------------------- security
    JWT_SECRET_KEY: str = Field(default="CHANGE_ME_IN_PRODUCTION")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # ---------------------------------------------------------------- database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/ai_sap_assistant"
    DATABASE_ECHO: bool = False

    # ---------------------------------------------------------------- redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---------------------------------------------------------------- LLM provider switch
    # This single value is the whole point of the abstraction: change it
    # (or override via env var) and the entire app talks to a different
    # LLM with zero code changes.
    LLM_PROVIDER: Literal["groq", "gemini", "huggingface", "ollama"] = "groq"

    # Groq (free, OpenAI-compatible, very fast — https://console.groq.com)
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # Google Gemini (free tier — https://aistudio.google.com)
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # HuggingFace Inference API (free tier — https://huggingface.co/settings/tokens)
    HF_API_TOKEN: str | None = None
    HF_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Ollama (fully local/free, no API key — https://ollama.com)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"

    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: int = 60

    # ---------------------------------------------------------------- embeddings
    # Local, free, offline — no API key needed. Runs on CPU via sentence-transformers.
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # ---------------------------------------------------------------- vector store
    VECTOR_STORE_PROVIDER: Literal["faiss", "chroma", "pgvector"] = "faiss"
    VECTOR_STORE_DIR: str = "./data/vector_store"
    VECTOR_DIMENSION: int = 1024  # bge-large-en-v1.5 output dimension

    # Chroma (local, free, persistent — https://www.trychroma.com)
    CHROMA_COLLECTION_NAME: str = "ai_sap_knowledge_assistant"

    # pgvector — reuses DATABASE_URL's host/credentials via a dedicated sync
    # (psycopg2) connection, kept separate from the app's async SQLAlchemy
    # engine so a slow vector query never ties up the async connection pool.
    PGVECTOR_TABLE_NAME: str = "vector_chunks"

    # ---------------------------------------------------------------- RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    RETRIEVAL_TOP_K: int = 8
    RERANK_TOP_K: int = 4
    HYBRID_DENSE_WEIGHT: float = 0.6
    HYBRID_SPARSE_WEIGHT: float = 0.4

    # ---------------------------------------------------------------- SAP
    SAP_MODE: Literal["mock", "real"] = "mock"
    SAP_GATEWAY_BASE_URL: str | None = None
    SAP_TOKEN_URL: str | None = None
    SAP_CLIENT_ID: str | None = None
    SAP_CLIENT_SECRET: str | None = None
    SAP_ODATA_MATERIAL_ENTITY_SET: str = "MaterialSet"
    SAP_ODATA_VENDOR_ENTITY_SET: str = "VendorSet"
    SAP_ODATA_PO_ENTITY_SET: str = "PurchaseOrderSet"
    SAP_ODATA_INVOICE_ENTITY_SET: str = "InvoiceSet"
    SAP_HTTP_TIMEOUT_SECONDS: int = 30

    # ---------------------------------------------------------------- background processing
    # "sync" ingests documents inline within the HTTP request (simplest, used
    # by default and by the test suite). "async" enqueues a Celery task and
    # returns immediately with status="processing" — needed once uploads are
    # large/frequent enough that blocking the request thread is unacceptable.
    DOCUMENT_PROCESSING_MODE: Literal["sync", "async"] = "sync"
    CELERY_BROKER_URL: str | None = None  # defaults to REDIS_URL if unset
    CELERY_RESULT_BACKEND: str | None = None  # defaults to REDIS_URL if unset

    # ---------------------------------------------------------------- observability
    ENABLE_PROMETHEUS: bool = True
    OTEL_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_SERVICE_NAME: str = "ai-sap-knowledge-assistant"

    # ---------------------------------------------------------------- audit
    AUDIT_LOG_MUTATING_REQUESTS: bool = True

    # ---------------------------------------------------------------- security controls
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".csv", ".md", ".html")

    # ---------------------------------------------------------------- cors
    CORS_ORIGINS: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")


@lru_cache
def get_settings() -> Settings:
    """Settings are cached — read once, reused everywhere (cheap DI)."""
    return Settings()
