from __future__ import annotations

from typing import Any

import httpx

from ...config import Settings
from ...config import load_settings
from ..base import LLMProvider
from ..types import LLMResponse
from ..types import Message
from ..types import ToolSpec


class OllamaProvider(LLMProvider):
    name: str = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._client = httpx.Client(base_url=self.settings.ollama_base_url, timeout=120.0)

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
        # Map our messages to Ollama's schema
        ollama_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in {"system", "user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model or "llama3.1",
            "messages": ollama_messages,
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        if top_p is not None:
            payload.setdefault("options", {})["top_p"] = top_p
        if max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = max_tokens

        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        content = message.get("content") or ""
        return LLMResponse(content=content, raw=data)

    def list_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        except Exception:
            # Fallback to common models
            return [
                "llama3.1",
                "llama3",
                "mistral",
                "mixtral",
                "phi3",
                "codellama",
            ]

    def list_chat_models(self) -> list[str]:
        """Get list of chat models (all Ollama models are chat models)."""
        # Ollama models are chat models, but most don't support tool calling
        # For agent use, we should filter to models that likely support it
        all_models = self.list_models()
        # Ollama models typically support basic chat, but tool calling support varies
        # For now, return all models (user can select)
        return all_models

    def list_embedding_models(self) -> list[str]:
        """Get list of embedding models."""
        # Ollama doesn't typically have separate embedding models
        # Embeddings are usually handled by the chat models themselves
        return []
