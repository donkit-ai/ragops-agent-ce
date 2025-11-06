from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from openai import AzureOpenAI

from ...config import Settings
from ...config import load_settings
from ..base import LLMProvider
from ..types import LLMResponse
from ..types import Message
from ..types import ToolCall
from ..types import ToolFunctionCall
from ..types import ToolSpec


class AzureOpenAIProvider(LLMProvider):
    name: str = "azure_openai"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

        if not self.settings.azure_openai_api_key:
            raise ValueError("RAGOPS_AZURE_OPENAI_API_KEY is not set")
        if not self.settings.azure_openai_endpoint:
            raise ValueError("RAGOPS_AZURE_OPENAI_ENDPOINT is not set")
        if not self.settings.azure_openai_deployment:
            raise ValueError("RAGOPS_AZURE_OPENAI_DEPLOYMENT is not set")

        self._client = AzureOpenAI(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
        )
        self._deployment = self.settings.azure_openai_deployment

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        """Get list of available models from Azure OpenAI API."""
        try:
            models = self._client.models.list()
            model_names = [model.id for model in models]
            return sorted(model_names, reverse=True)
        except Exception:
            # Fallback to common models
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]

    def list_chat_models(self) -> list[str]:
        """Get list of chat models with tool calling support."""
        try:
            models = self._client.models.list()
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
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ]

    def list_embedding_models(self) -> list[str]:
        """Get list of embedding models."""
        try:
            models = self._client.models.list()
            model_names = []
            for model in models:
                name = model.id
                if "embedding" in name.lower():
                    model_names.append(name)
            return sorted(model_names, reverse=True)
        except Exception:
            return [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]

    def _serialize_message(self, message: Message) -> dict[str, Any]:
        """Serialize message for Azure OpenAI API (arguments must be JSON string)."""
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
        # Use deployment from settings or override with model parameter
        deployment = model or self._deployment

        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": [self._serialize_message(m) for m in messages],
        }

        if tools:
            kwargs["tools"] = [t.model_dump(exclude_none=True) for t in tools]
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0].message
        content: str = choice.content or ""
        tool_calls_data = choice.tool_calls or []
        tool_calls: list[ToolCall] | None = None

        if tool_calls_data:
            tool_calls = []
            for tc in tool_calls_data:
                # Parse JSON string to dict
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        function=ToolFunctionCall(
                            name=tc.function.name,
                            arguments=args,
                        ),
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls, raw=response.model_dump())

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
        # Use deployment from settings or override with model parameter
        deployment = model or self._deployment

        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": [self._serialize_message(m) for m in messages],
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
            kwargs["max_tokens"] = max_tokens

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
