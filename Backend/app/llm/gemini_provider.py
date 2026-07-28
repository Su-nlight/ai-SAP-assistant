"""
Google Gemini provider — free tier via Google AI Studio REST API.
https://aistudio.google.com

Gemini uses "user"/"model" roles and a separate `systemInstruction` field
rather than a "system" chat message, so this provider translates our
normalized `ChatMessage` list into Gemini's shape internally. That
translation detail never leaks above this file.
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.base import ChatMessage, ChatRole, LLMProvider, LLMProviderError, LLMResponse

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60) -> None:
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is not configured.")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    @staticmethod
    def _split_system_and_turns(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        system_parts: list[str] = []
        turns: list[dict] = []
        for m in messages:
            if m.role == ChatRole.SYSTEM:
                system_parts.append(m.content)
            else:
                gemini_role = "model" if m.role == ChatRole.ASSISTANT else "user"
                turns.append({"role": gemini_role, "parts": [{"text": m.content}]})
        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, turns

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
        system_instruction, turns = self._split_system_and_turns(messages)

        payload: dict = {
            "contents": turns,
            "generationConfig": {
                "temperature": temperature if temperature is not None else 0.2,
                "maxOutputTokens": max_tokens or 1024,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{_GEMINI_BASE_URL}/{self._model}:generateContent?key={self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(f"Gemini API error {exc.response.status_code}: {exc.response.text}") from exc
            except httpx.TransportError as exc:
                raise LLMProviderError(f"Gemini API unreachable: {exc}") from exc

        data = resp.json()
        latency_ms = (self._timed() - started) * 1000

        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMProviderError(f"Unexpected Gemini response shape: {data}") from exc

        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
            latency_ms=latency_ms,
            raw=data,
        )