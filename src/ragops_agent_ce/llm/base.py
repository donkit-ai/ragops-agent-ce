from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from .types import LLMResponse, Message, ToolSpec


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
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
        raise NotImplementedError

    def generate_stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[LLMResponse]:
        """Stream generation. Yields partial responses.

        Override in subclass for streaming support.
        """
        raise NotImplementedError(f"{self.name} provider does not support streaming")

    def supports_tools(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        return False
