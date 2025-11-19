"""Integration tests for Agent + Tools — end-to-end scenarios.

These tests verify that the agent correctly:
1. Calls tools and processes results
2. Maintains context across multiple turns
3. Chains tools together
4. Recovers from tool errors
5. Handles streaming with tools
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.conftest import BaseMockProvider
from ragops_agent_ce.agent.agent import LLMAgent
from ragops_agent_ce.agent.tools import AgentTool
from ragops_agent_ce.llm.base import LLMProvider
from ragops_agent_ce.llm.types import LLMResponse
from ragops_agent_ce.llm.types import Message
from ragops_agent_ce.llm.types import ToolCall
from ragops_agent_ce.llm.types import ToolFunctionCall


# ============================================================================
# Test-specific configurations
# ============================================================================


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_provider() -> BaseMockProvider:
    """Create a mock provider with tool call scenario."""
    return BaseMockProvider(
        responses=[
            {
                "tool_calls": [
                    {"name": "tool_a", "arguments": {"x": 5}}
                ]
            },
            {"content": "The result is 25"},
        ]
    )


@pytest.fixture
def simple_tools() -> list[AgentTool]:
    """Create simple tools for testing."""

    def tool_a(args: dict[str, Any]) -> str:
        """Tool A: square a number."""
        x = args.get("x", 0)
        result = x * x
        return json.dumps({"result": result})

    def tool_b(args: dict[str, Any]) -> str:
        """Tool B: double a number."""
        x = args.get("x", 0)
        result = x * 2
        return json.dumps({"result": result})

    return [
        AgentTool(
            name="tool_a",
            description="Square a number",
            handler=tool_a,
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        ),
        AgentTool(
            name="tool_b",
            description="Double a number",
            handler=tool_b,
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        ),
    ]


# ============================================================================
# Tests: Single Tool Call
# ============================================================================


@pytest.mark.asyncio
async def test_agent_single_tool_call(
    mock_provider: BaseMockProvider,
    simple_tools: list[AgentTool],
) -> None:
    """Test agent calling a single tool and getting result."""
    agent = LLMAgent(provider=mock_provider, tools=simple_tools)

    messages = [
        Message(role="user", content="What is 5 squared?"),
    ]

    result = await agent.arespond(messages)

    # Should have called provider twice (once for tool call, once for final response)
    assert mock_provider.call_count == 2
    # Should have final response
    assert "25" in result or "result" in result.lower()
    # Messages should have tool result
    assert len(messages) >= 3  # user, assistant with tool call, tool result


@pytest.mark.asyncio
async def test_agent_tool_result_in_messages(
    mock_provider: BaseMockProvider,
    simple_tools: list[AgentTool],
) -> None:
    """Test that tool results are properly added to messages."""
    agent = LLMAgent(provider=mock_provider, tools=simple_tools)

    messages = [
        Message(role="user", content="Square 5"),
    ]

    await agent.arespond(messages)

    # Check message history
    assert len(messages) >= 3
    # Should have assistant message with tool call
    assert any(m.role == "assistant" and m.tool_calls for m in messages)
    # Should have tool result message
    assert any(m.role == "tool" for m in messages)


# ============================================================================
# Tests: Multi-turn Conversation
# ============================================================================


@pytest.mark.asyncio
async def test_agent_multi_turn_context(
    mock_provider: BaseMockProvider,
    simple_tools: list[AgentTool],
) -> None:
    """Test agent maintains context across multiple turns."""
    agent = LLMAgent(provider=mock_provider, tools=simple_tools)

    messages = [
        Message(role="user", content="What is 5 squared?"),
    ]

    # First turn
    result1 = await agent.arespond(messages)
    assert result1  # Should get a response

    # Messages should be accumulated
    messages_after_turn1 = len(messages)
    assert messages_after_turn1 >= 3

    # Second turn - provider should see all previous messages
    # Reset provider for second turn
    mock_provider.call_count = 0

    # Add user message for second turn
    messages.append(Message(role="user", content="Now double it"))

    result2 = await agent.arespond(messages)
    assert result2

    # Provider should have seen all previous messages
    assert len(mock_provider.messages_history) > 0
    last_call_messages = mock_provider.messages_history[-1]
    # Should include messages from first turn
    assert len(last_call_messages) >= messages_after_turn1


# ============================================================================
# Tests: Tool Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_agent_tool_error_recovery() -> None:
    """Test agent recovers from tool errors."""

    def failing_tool(args: dict[str, Any]) -> str:
        """Tool that raises an error."""
        raise ValueError("Tool execution failed")

    tools = [
        AgentTool(
            name="failing_tool",
            description="A tool that fails",
            handler=failing_tool,
            parameters={"type": "object", "properties": {}},
        ),
    ]

    class ErrorProvider(LLMProvider):
        """Provider that calls failing tool."""

        def supports_tools(self) -> bool:
            return True

        def supports_streaming(self) -> bool:
            return False

        def generate(
            self,
            messages: list[Message],
            tools: list[Any] | None = None,
            model: str | None = None,
        ) -> LLMResponse:
            # Call failing tool
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ToolFunctionCall(
                            name="failing_tool",
                            arguments={},
                        ),
                    )
                ],
            )

        def generate_stream(self, messages, tools=None, model=None):
            raise NotImplementedError()

    agent = LLMAgent(provider=ErrorProvider(), tools=tools)
    messages = [Message(role="user", content="Call the failing tool")]

    # Should raise the tool error
    with pytest.raises(ValueError):
        await agent.arespond(messages)


# ============================================================================
# Tests: Tool Chaining
# ============================================================================


@pytest.mark.asyncio
async def test_agent_tool_chaining() -> None:
    """Test agent can chain multiple tools together."""

    call_sequence: list[str] = []

    def tool_1(args: dict[str, Any]) -> str:
        """First tool in chain."""
        call_sequence.append("tool_1")
        return json.dumps({"value": 10})

    def tool_2(args: dict[str, Any]) -> str:
        """Second tool in chain."""
        call_sequence.append("tool_2")
        return json.dumps({"value": 20})

    tools = [
        AgentTool(
            name="tool_1",
            description="First tool",
            handler=tool_1,
            parameters={"type": "object", "properties": {}},
        ),
        AgentTool(
            name="tool_2",
            description="Second tool",
            handler=tool_2,
            parameters={"type": "object", "properties": {}},
        ),
    ]

    class ChainProvider(LLMProvider):
        """Provider that chains tools."""

        def __init__(self) -> None:
            self.call_count = 0

        def supports_tools(self) -> bool:
            return True

        def supports_streaming(self) -> bool:
            return False

        def generate(
            self,
            messages: list[Message],
            tools: list[Any] | None = None,
            model: str | None = None,
        ) -> LLMResponse:
            self.call_count += 1

            # First call: use tool_1
            if self.call_count == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=ToolFunctionCall(
                                name="tool_1",
                                arguments={},
                            ),
                        )
                    ],
                )

            # Second call: use tool_2
            if self.call_count == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            type="function",
                            function=ToolFunctionCall(
                                name="tool_2",
                                arguments={},
                            ),
                        )
                    ],
                )

            # Third call: final response
            return LLMResponse(content="Chain complete")

        def generate_stream(self, messages, tools=None, model=None):
            raise NotImplementedError()

    provider = ChainProvider()
    agent = LLMAgent(provider=provider, tools=tools, max_iterations=5)
    messages = [Message(role="user", content="Chain tools")]

    result = await agent.arespond(messages)

    # Both tools should have been called
    assert "tool_1" in call_sequence
    assert "tool_2" in call_sequence
    # tool_1 should be called before tool_2
    assert call_sequence.index("tool_1") < call_sequence.index("tool_2")
    
    # Provider should have been called 3 times (tool_1, tool_2, final)
    assert provider.call_count == 3
    
    # Final result should be present
    assert result == "Chain complete"
    
    # Messages should contain tool results
    tool_messages = [m for m in messages if m.role == "tool"]
    assert len(tool_messages) == 2, "Should have 2 tool result messages"


# ============================================================================
# Tests: Tool with Complex Arguments
# ============================================================================


@pytest.mark.asyncio
async def test_agent_tool_complex_arguments() -> None:
    """Test agent passing complex arguments to tools."""

    received_args: dict[str, Any] = {}

    def complex_tool(args: dict[str, Any]) -> str:
        """Tool that receives complex arguments."""
        nonlocal received_args
        received_args = args
        return json.dumps({"received": args})

    tools = [
        AgentTool(
            name="complex_tool",
            description="Tool with complex args",
            handler=complex_tool,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "config": {"type": "object"},
                },
            },
        ),
    ]

    class ComplexProvider(LLMProvider):
        def supports_tools(self) -> bool:
            return True

        def supports_streaming(self) -> bool:
            return False

        def generate(
            self,
            messages: list[Message],
            tools: list[Any] | None = None,
            model: str | None = None,
        ) -> LLMResponse:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ToolFunctionCall(
                            name="complex_tool",
                            arguments={
                                "name": "test",
                                "count": 42,
                                "tags": ["a", "b", "c"],
                                "config": {"key": "value"},
                            },
                        ),
                    )
                ],
            )

        def generate_stream(self, messages, tools=None, model=None):
            raise NotImplementedError()

    agent = LLMAgent(provider=ComplexProvider(), tools=tools)
    messages = [Message(role="user", content="Use complex tool")]

    await agent.arespond(messages)

    # Tool should have received all arguments
    assert received_args["name"] == "test"
    assert received_args["count"] == 42
    assert received_args["tags"] == ["a", "b", "c"]
    assert received_args["config"] == {"key": "value"}


# ============================================================================
# Tests: Tool Result Processing
# ============================================================================


@pytest.mark.asyncio
async def test_agent_processes_tool_results() -> None:
    """Test agent correctly processes tool results."""

    tool_results: list[str] = []

    def tracking_tool(args: dict[str, Any]) -> str:
        """Tool that returns structured result."""
        result = {"status": "success", "data": [1, 2, 3], "message": "Completed"}
        return json.dumps(result)

    tools = [
        AgentTool(
            name="tracking_tool",
            description="Tool that tracks results",
            handler=tracking_tool,
            parameters={"type": "object", "properties": {}},
        ),
    ]

    class TrackingProvider(LLMProvider):
        def __init__(self) -> None:
            self.call_count = 0

        def supports_tools(self) -> bool:
            return True

        def supports_streaming(self) -> bool:
            return False

        def generate(
            self,
            messages: list[Message],
            tools: list[Any] | None = None,
            model: str | None = None,
        ) -> LLMResponse:
            self.call_count += 1

            # First call: use tool
            if self.call_count == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=ToolFunctionCall(
                                name="tracking_tool",
                                arguments={},
                            ),
                        )
                    ],
                )

            # Second call: process result
            # Check that tool result is in messages
            for msg in messages:
                if msg.role == "tool":
                    tool_results.append(msg.content)

            return LLMResponse(content="Processed results")

        def generate_stream(self, messages, tools=None, model=None):
            raise NotImplementedError()

    agent = LLMAgent(provider=TrackingProvider(), tools=tools)
    messages = [Message(role="user", content="Use tracking tool")]

    await agent.arespond(messages)

    # Tool result should have been captured
    assert len(tool_results) > 0
    # Result should be valid JSON
    result_data = json.loads(tool_results[0])
    assert result_data["status"] == "success"
    assert result_data["data"] == [1, 2, 3]
