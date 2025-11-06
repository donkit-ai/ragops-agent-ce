from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator

from .types import LLMResponse
from .types import Message
from .types import ToolSpec


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

    def list_models(self) -> list[str]:
        """Get list of available models from the provider.

        Returns:
            List of model names available from this provider.
            If listing is not supported, returns empty list.
        """
        return []

    def list_chat_models(self) -> list[str]:
        """Get list of chat models suitable for agent use (with tool calling support).

        Returns:
            List of model names that support chat completions and tool calling.
            Filters out embedding-only models and models without tool support.
        """
        # Default: filter from all models, keeping only those that support tools
        all_models = self.list_models()
        if self.supports_tools():
            # If provider supports tools, assume all models support them
            # Filter out known embedding-only models
            embedding_models = {
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            }
            return [m for m in all_models if not any(emb in m.lower() for emb in embedding_models)]
        return []

    def list_embedding_models(self) -> list[str]:
        """Get list of embedding models suitable for RAG.

        Returns:
            List of model names that support embeddings.
        """
        # Default: filter from all models, keeping only embedding models
        all_models = self.list_models()
        embedding_keywords = {
            "embedding",
            "embed",
        }
        # Also include known embedding models
        known_embedding_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        embedding_models = [
            m
            for m in all_models
            if any(kw in m.lower() for kw in embedding_keywords) or m in known_embedding_models
        ]
        # If no embedding models found but provider supports embeddings, return common ones
        if not embedding_models and self.name in ("openai", "azure_openai", "vertex"):
            return list(known_embedding_models)
        return embedding_models
