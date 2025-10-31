from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str | None = None  # Can be None for assistant with tool_calls
    name: str | None = None  # Only for role="tool"
    tool_call_id: str | None = None  # Only for role="tool"
    tool_calls: list[ToolCall] | None = None  # Only for role="assistant"


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ToolFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]  # Parsed from JSON string by providers


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunctionCall


class LLMResponse(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw: Any | None = None

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls,
        )
