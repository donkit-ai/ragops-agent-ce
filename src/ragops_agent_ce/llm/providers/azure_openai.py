from __future__ import annotations

import json
from typing import Any

from openai import AzureOpenAI

from ...config import Settings, load_settings
from ..base import LLMProvider
from ..types import LLMResponse, Message, ToolCall, ToolFunctionCall, ToolSpec


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
