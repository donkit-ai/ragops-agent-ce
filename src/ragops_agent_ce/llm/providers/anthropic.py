from __future__ import annotations

from typing import Any

import httpx

from ...config import Settings, load_settings
from ..base import LLMProvider
from ..types import LLMResponse, Message, ToolSpec


class AnthropicProvider(LLMProvider):
    name: str = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        if not self.settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            timeout=60.0,
        )

    def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        # Anthropic expects a single system prompt and a list of user/assistant messages
        system_msgs = [m.content for m in messages if m.role == "system"]
        system = "\n\n".join(system_msgs) if system_msgs else None
        chat_messages = [m for m in messages if m.role in {"user", "assistant"}]

        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model or "claude-3-5-sonnet-latest",
            "messages": [{"role": m.role, "content": m.content} for m in chat_messages],
            "max_tokens": max_tokens or 1024,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        resp = self._client.post("/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        content = "\n".join([t for t in text_parts if t])
        return LLMResponse(content=content, raw=data)
