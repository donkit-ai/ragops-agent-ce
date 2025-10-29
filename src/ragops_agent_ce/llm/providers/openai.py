from __future__ import annotations

import json
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
