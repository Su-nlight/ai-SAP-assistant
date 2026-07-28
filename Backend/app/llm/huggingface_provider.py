"""
HuggingFace Inference API provider — free tier chat-completions endpoint.
https://huggingface.co/settings/tokens

Uses the OpenAI-compatible router HuggingFace exposes for chat models
(`https://router.huggingface.co/v1/chat/completions`), so the request/response
shape mirrors Groq/OpenAI and needs no special translation.
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.base import ChatMessage, LLMProvider, LLMProviderError, LLMResponse

_HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceProvider(LLMProvider):
    provider_name = "huggingface"

    def __init__(self, api_token: str, model: str, timeout_seconds: int = 60) -> None:
        if not api_token:
            raise LLMProviderError("HF_API_TOKEN is not configured.")
        self._api_token = api_token
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
        headers = {"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(_HF_ROUTER_URL, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(f"HuggingFace API error {exc.response.status_code}: {exc.response.text}") from exc
            except httpx.TransportError as exc:
                raise LLMProviderError(f"HuggingFace API unreachable: {exc}") from exc

        data = resp.json()
        latency_ms = (self._timed() - started) * 1000
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=latency_ms,
            raw=data,
        )
