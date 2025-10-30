from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from ...config import Settings, load_settings
from ..base import LLMProvider
from ..types import LLMResponse, Message, ToolCall, ToolFunctionCall, ToolSpec


class OpenAIProvider(LLMProvider):
    name: str = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=60.0,
        )
        self._model_name = model_name

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

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
