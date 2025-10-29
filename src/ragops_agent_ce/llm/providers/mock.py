from __future__ import annotations

from ...config import Settings, load_settings
from ..base import LLMProvider
from ..types import LLMResponse, Message, ToolSpec


class MockProvider(LLMProvider):
    name: str = "mock"

    def __init__(self, settings: Settings | None = None):
        # Store settings for consistency with other providers
        self.settings = settings or load_settings()

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
        # Simple echo of the last user message
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        content = f"[mock]{' ' + last_user.content if last_user else ''}".strip()
        return LLMResponse(content=content, raw={"provider": "mock"})
