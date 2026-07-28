"""
LLM Provider Factory.

This is the single place in the whole codebase that knows how many LLM
providers exist and how to build one. Everything else — RAG pipeline,
chat service, tests — asks for `get_llm_provider(settings)` and receives
an `LLMProvider`, never a concrete class.

To add a new provider (e.g. Azure OpenAI):
  1. Implement `LLMProvider` in a new `azure_openai_provider.py`.
  2. Add one branch here.
  3. Add its settings to `core/config.py`.
No other file changes.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.huggingface_provider import HuggingFaceProvider
from app.llm.ollama_provider import OllamaProvider


def _build_provider(settings: Settings) -> LLMProvider:
    match settings.LLM_PROVIDER:
        case "groq":
            return GroqProvider(
                api_key=settings.GROQ_API_KEY or "",
                model=settings.GROQ_MODEL,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
        case "gemini":
            return GeminiProvider(
                api_key=settings.GEMINI_API_KEY or "",
                model=settings.GEMINI_MODEL,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
        case "huggingface":
            return HuggingFaceProvider(
                api_token=settings.HF_API_TOKEN or "",
                model=settings.HF_MODEL,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
        case "ollama":
            return OllamaProvider(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
        case other:
            raise LLMProviderError(f"Unknown LLM_PROVIDER '{other}'.")


@lru_cache
def get_llm_provider() -> LLMProvider:
    """
    Cached singleton so we don't rebuild a provider per request.

    Takes no arguments deliberately: `Settings` (pydantic) isn't hashable,
    so it can't be an `lru_cache` key. We instead pull the already-cached
    settings singleton internally — there is exactly one active provider
    per process, which is what we want.
    """
    return _build_provider(get_settings())


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Uncached variant for tests that need to construct a provider from
    a specific, non-singleton `Settings` instance (e.g. with monkeypatched
    env vars)."""
    return _build_provider(settings)
