from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from ...config import Settings
from ...config import load_settings
from ..base import LLMProvider
from ..types import LLMResponse
from ..types import Message
from ..types import ToolCall
from ..types import ToolFunctionCall
from ..types import ToolSpec


class OpenAIProvider(LLMProvider):
    name: str = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        # API key is required for real OpenAI, but not for Ollama/OpenRouter
        api_key = self.settings.openai_api_key or "not-needed"
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.settings.openai_base_url,
            timeout=60.0,
        )
        self._model_name = model_name

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        """Get list of available models from OpenAI API."""
        try:
            models = self._client.models.list()
            # Include all models (chat and embedding)
            model_names = [model.id for model in models]
            return sorted(model_names, reverse=True)
        except Exception:
            # If API call fails, return common models as fallback
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]

    def list_chat_models(self) -> list[str]:
        """Get list of chat models with tool calling support.

        Note: models.list() returns models available for your API key,
        but final availability is validated when a model is selected.
        Some models may be restricted by region, subscription plan, or early access.
        """
        try:
            models = self._client.models.list()
            # Filter to chat completion models that support function calling
            # GPT models and o1 models support tool calling
            model_names = []
            for model in models:
                name = model.id
                # Exclude embedding models
                if "embedding" in name.lower():
                    continue
                # Include GPT models and o1 models
                if name.startswith("gpt-") or name.startswith("o1-"):
                    model_names.append(name)
            return sorted(model_names, reverse=True)
        except Exception:
            # Fallback to common chat models
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini",
            ]

    def list_embedding_models(self) -> list[str]:
        """Get list of embedding models."""
        try:
            models = self._client.models.list()
            # Filter to embedding models
            model_names = []
            for model in models:
                name = model.id
                if "embedding" in name.lower():
                    model_names.append(name)
            return sorted(model_names, reverse=True)
        except Exception:
            # Fallback to common embedding models
            return [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]

    def _serialize_message(self, message: Message) -> dict[str, Any]:
        """Serialize message for OpenAI API (arguments must be JSON string)."""
        msg_dict = message.model_dump(exclude_none=True)

        # Convert tool_calls arguments from dict to JSON string
        if msg_dict.get("tool_calls"):
            for tc in msg_dict["tool_calls"]:
                if "function" in tc and "arguments" in tc["function"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, dict):
                        tc["function"]["arguments"] = json.dumps(args)

        return msg_dict

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
        # Convert messages to OpenAI format
        openai_messages = [self._serialize_message(m) for m in messages]

        # Prepare kwargs for the API call
        kwargs = {
            "model": model or self._model_name or self.settings.llm_model or "gpt-4o-mini",
            "messages": openai_messages,
        }

        if tools:
            kwargs["tools"] = [t.model_dump(exclude_none=True) for t in tools]
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            if "gpt" in model and "oss" not in model:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        # Make API call
        response = self._client.chat.completions.create(**kwargs)

        # Extract response data
        choice = response.choices[0].message
        content: str = choice.content or ""

        # Parse tool calls if present
        tool_calls: list[ToolCall] | None = None
        if choice.tool_calls:
            tool_calls = []
            for tc in choice.tool_calls:
                # Parse JSON string to dict
                args_str = tc.function.arguments
                args = json.loads(args_str) if args_str else {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        function=ToolFunctionCall(name=tc.function.name, arguments=args),
                    )
                )

        # Convert response to dict for raw field
        raw = response.model_dump()

        return LLMResponse(content=content, tool_calls=tool_calls, raw=raw)

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
        """Stream generation yielding partial responses."""
        # Convert messages to OpenAI format
        openai_messages = [self._serialize_message(m) for m in messages]

        # Prepare kwargs for the API call
        kwargs = {
            "model": model or self._model_name or self.settings.llm_model or "gpt-4o-mini",
            "messages": openai_messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = [t.model_dump(exclude_none=True) for t in tools]
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        # Make streaming API call
        stream = self._client.chat.completions.create(**kwargs)

        # Accumulate tool calls across chunks
        accumulated_tool_calls: dict[int, dict] = {}

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Yield text content if present
            if delta.content:
                yield LLMResponse(content=delta.content, tool_calls=None)

            # Accumulate tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name": "", "arguments": ""},
                        }

                    if tc_delta.id:
                        accumulated_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[idx]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

        # Yield final response with accumulated tool calls if any
        if accumulated_tool_calls:
            tool_calls = []
            for tc_data in accumulated_tool_calls.values():
                args_str = tc_data["function"]["arguments"]
                args = json.loads(args_str) if args_str else {}
                tool_calls.append(
                    ToolCall(
                        id=tc_data["id"],
                        function=ToolFunctionCall(
                            name=tc_data["function"]["name"],
                            arguments=args,
                        ),
                    )
                )
            yield LLMResponse(content=None, tool_calls=tool_calls)
