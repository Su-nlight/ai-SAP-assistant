"""
Ollama provider — fully local, fully free, no API key, no external network
dependency. Requires `ollama serve` running locally with the model pulled
(`ollama pull llama3.1`). Ideal for offline development or air-gapped demos.
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.base import ChatMessage, LLMProvider, LLMProviderError, LLMResponse


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(httpx.TransportError),
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
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else 0.2,
                "num_predict": max_tokens or 1024,
            },
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(f"Ollama error {exc.response.status_code}: {exc.response.text}") from exc
            except httpx.TransportError as exc:
                raise LLMProviderError(
                    f"Cannot reach Ollama at {self._base_url}. Is `ollama serve` running? ({exc})"
                ) from exc

        data = resp.json()
        latency_ms = (self._timed() - started) * 1000
        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0) or None,
            latency_ms=latency_ms,
            raw=data,
        )
