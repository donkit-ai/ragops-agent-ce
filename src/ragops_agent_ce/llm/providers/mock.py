from __future__ import annotations

from collections.abc import AsyncIterator

from donkit.llm import (
    GenerateRequest,
    GenerateResponse,
    LLMModelAbstract,
    ModelCapability,
    StreamChunk,
)

from ...config import Settings, load_settings


class MockProvider(LLMModelAbstract):
    """Mock LLM provider for testing."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._model_name = "mock-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value

    @property
    def capabilities(self) -> ModelCapability:
        return ModelCapability.TEXT_GENERATION | ModelCapability.STREAMING

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # Simple echo of the last user message
        last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
        content = (
            f"[mock]{' ' + last_user.content if last_user and last_user.content else ''}".strip()
        )
        return GenerateResponse(
            content=content,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        # Stream the mock response word by word
        last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
        content = (
            f"[mock]{' ' + last_user.content if last_user and last_user.content else ''}".strip()
        )

        words = content.split()
        for i, word in enumerate(words):
            yield StreamChunk(
                content=word + (" " if i < len(words) - 1 else ""),
                finish_reason="stop" if i == len(words) - 1 else None,
            )
