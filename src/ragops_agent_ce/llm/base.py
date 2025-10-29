from __future__ import annotations

from abc import ABC, abstractmethod

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

    def supports_tools(self) -> bool:
        return False
