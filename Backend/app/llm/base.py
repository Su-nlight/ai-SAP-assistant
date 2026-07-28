"""
LLM Provider abstraction (Strategy pattern).

Every concrete provider (Groq, Gemini, HuggingFace, Ollama, ...) implements
this interface. Nothing above this layer (RAG pipeline, chat service) ever
imports a concrete provider — it depends only on `LLMProvider`. Swapping
models is a one-line config change (`LLM_PROVIDER=...`), never a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(slots=True)
class LLMResponse:
    """Normalized response shape — identical regardless of which provider answered."""

    content: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Raised when an upstream LLM call fails after retries."""


class LLMProvider(ABC):
    """Common interface every LLM backend must satisfy."""

    provider_name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat-style conversation and return a normalized response."""
        raise NotImplementedError

    @staticmethod
    def _timed() -> Any:
        """Small helper so subclasses measure latency consistently."""
        return perf_counter()
