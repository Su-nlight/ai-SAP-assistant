"""
Groq provider — free, extremely fast LLM inference (Llama 3.1, Mixtral, etc.)
via an OpenAI-compatible REST API. https://console.groq.com
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.base import ChatMessage, LLMProvider, LLMProviderError, LLMResponse

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(LLMProvider):
    provider_name = "groq"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60) -> None:
        if not api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured.")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = self._timed()
        payload = {
            "model": self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else 0.2,
            "max_tokens": max_tokens or 1024,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(_GROQ_CHAT_URL, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(f"Groq API error {exc.response.status_code}: {exc.response.text}") from exc
            except httpx.TransportError as exc:
                raise LLMProviderError(f"Groq API unreachable: {exc}") from exc

        data = resp.json()
        latency_ms = (self._timed() - started) * 1000
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=latency_ms,
            raw=data,
        )
