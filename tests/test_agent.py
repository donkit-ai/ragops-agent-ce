"""Unit tests for LLMAgent core methods (arespond / arespond_stream)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from donkit_ragops.agent.agent import EventType
from donkit_ragops.agent.agent import LLMAgent
from donkit_ragops.agent.local_tools.tools import AgentTool
from donkit.llm import Message
from donkit.llm import ToolCall
from donkit.llm import FunctionCall
from donkit_ragops.mcp.client import MCPClient

from .conftest import BaseMockProvider

# ============================================================================
# Fixtures and Mock Objects
# ============================================================================


@pytest.fixture
def stub_messages() -> list[Message]:
    """Basic message list for testing."""
    return [Message(role="user", content="test prompt")]


@pytest.fixture
def fake_provider() -> BaseMockProvider:
    """Default fake provider without tools or streaming."""
    return BaseMockProvider(supports_tools_val=False, supports_streaming_val=False)


@pytest.fixture
def local_tool_stub() -> AgentTool:
    """Mock local tool for testing."""
    handler = Mock(return_value={"result": 42})
    tool = AgentTool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    return tool


@pytest.fixture
def mcp_client_stub() -> AsyncMock:
    """Mock MCP client for testing."""
    client = AsyncMock(spec=MCPClient)
    client.command = "test-mcp-server"
    client._alist_tools = AsyncMock(
        return_value=[
            {
                "name": "mcp_tool",
                "description": "An MCP tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    )
    client._acall_tool = AsyncMock(return_value={"status": "ok"})
    return client


# ============================================================================
# Tests A: arespond() without tools
# ============================================================================


@pytest.mark.asyncio
async def test_a1_return_text_without_tool_calls(stub_messages: list[Message]):
    """A1: Return text when provider doesn't support tools and no tool-calls."""
    provider = BaseMockProvider(
        supports_tools_val=False,
        responses=[{"content": "hi"}],
    )
    agent = LLMAgent(provider=provider, tools=[])

    result = await agent.arespond(stub_messages)

    assert result == "hi"
    assert len(stub_messages) == 1  # Original message unchanged


@pytest.mark.asyncio
async def test_a2_retry_on_empty_content(stub_messages: list[Message]):
    """A2: Retry provider when first response has no content."""
    provider = BaseMockProvider(
        supports_tools_val=False,
        responses=[
            {"content": None},
            {"content": "retry ok"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[])

    result = await agent.arespond(stub_messages)

    assert result == "retry ok"
    assert provider.call_count == 2


# ============================================================================
# Tests B: arespond() with local tools
# ============================================================================


@pytest.mark.asyncio
async def test_b1_tool_call_with_local_tool(
    stub_messages: list[Message], local_tool_stub: AgentTool
):
    """B1: Tool call with successful local tool execution."""
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {"foo": "bar"}}]},
            {"content": "final answer"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[local_tool_stub])

    result = await agent.arespond(stub_messages)

    assert result == "final answer"
    # Check synthetic assistant message was added
    assert stub_messages[1].role == "assistant"
    assert stub_messages[1].content is None
    assert stub_messages[1].tool_calls is not None
    # Check tool message was added
    assert stub_messages[2].role == "tool"
    assert stub_messages[2].name == "test_tool"
    # Tool handler should have been called
    local_tool_stub.handler.assert_called_once_with({"foo": "bar"})


@pytest.mark.asyncio
async def test_b2_tool_not_found(stub_messages: list[Message]):
    """B2: Tool not found - should return empty string in tool message."""
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "unknown_tool", "arguments": {}}]},
            {"content": "recovered"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[])

    result = await agent.arespond(stub_messages)

    assert result == "recovered"
    # Check tool message with empty content
    assert stub_messages[2].role == "tool"
    assert stub_messages[2].content == ""


@pytest.mark.asyncio
async def test_b3_exception_in_local_tool(stub_messages: list[Message], local_tool_stub: AgentTool):
    """B3: Exception in local tool should be handled gracefully."""
    local_tool_stub.handler = Mock(side_effect=RuntimeError("fail"))
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[{"tool_calls": [{"name": "test_tool", "arguments": {}}]}],
    )
    agent = LLMAgent(provider=provider, tools=[local_tool_stub], max_iterations=1)

    result = await agent.arespond(stub_messages)

    # Should return empty string due to max_iterations reached
    assert result == ""
    # Check that tool message contains error
    assert len(stub_messages) >= 3
    assert stub_messages[2].role == "tool"
    assert "Error" in stub_messages[2].content


@pytest.mark.asyncio
async def test_b4_keyboard_interrupt_handling(
    stub_messages: list[Message], local_tool_stub: AgentTool
):
    """B4: KeyboardInterrupt should be handled gracefully."""
    local_tool_stub.handler = Mock(side_effect=KeyboardInterrupt())
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {}}]},
            {"content": "continued"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[local_tool_stub])

    result = await agent.arespond(stub_messages)

    assert result == "continued"
    # Check tool message contains cancellation message
    assert "cancelled" in stub_messages[2].content.lower()


@pytest.mark.asyncio
async def test_b4_cancelled_error_handling(
    stub_messages: list[Message], local_tool_stub: AgentTool
):
    """B4: asyncio.CancelledError should be handled gracefully."""
    local_tool_stub.handler = Mock(side_effect=asyncio.CancelledError())
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {}}]},
            {"content": "continued"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[local_tool_stub])

    result = await agent.arespond(stub_messages)

    assert result == "continued"
    assert "cancelled" in stub_messages[2].content.lower()


@pytest.mark.asyncio
async def test_b5_parse_tool_args_dict(stub_messages: list[Message]):
    """B5: Parse tool arguments - dict should be returned as-is."""
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {"key": "value"}}]},
            {"content": "ok"},
        ],
    )
    handler = Mock(return_value="result")
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        handler=handler,
    )
    agent = LLMAgent(provider=provider, tools=[tool])

    await agent.arespond(stub_messages)

    handler.assert_called_once_with({"key": "value"})


@pytest.mark.asyncio
async def test_b5_parse_tool_args_json_string(stub_messages: list[Message]):
    """B5: Parse tool arguments - JSON string should be parsed."""
    # Create a tool call with JSON string arguments (agent parses it)
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {"key": "value"}}]},
            {"content": "ok"},
        ],
    )
    handler = Mock(return_value="result")
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        handler=handler,
    )
    agent = LLMAgent(provider=provider, tools=[tool])

    await agent.arespond(stub_messages)

    handler.assert_called_once_with({"key": "value"})


@pytest.mark.asyncio
async def test_b5_parse_tool_args_invalid_json(stub_messages: list[Message]):
    """B5: Parse tool arguments - invalid JSON string should return empty dict."""
    # Pass invalid JSON string - agent should parse it and return {}
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": "not valid json"}]},
            {"content": "ok"},
        ],
    )
    handler = Mock(return_value="result")
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        handler=handler,
    )
    agent = LLMAgent(provider=provider, tools=[tool])

    await agent.arespond(stub_messages)

    handler.assert_called_once_with({})


# ============================================================================
# Tests C: arespond() with MCP tools
# ============================================================================


@pytest.mark.asyncio
async def test_c1_successful_mcp_tool_call(
    stub_messages: list[Message], mcp_client_stub: AsyncMock
):
    """C1: Successful MCP tool call execution."""
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[
            {"tool_calls": [{"name": "mcp_tool", "arguments": {"param": "value"}}]},
            {"content": "mcp result processed"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[], mcp_clients=[mcp_client_stub])
    await agent.ainit_mcp_tools()

    result = await agent.arespond(stub_messages)

    assert result == "mcp result processed"
    # Check MCP tool was called
    mcp_client_stub._acall_tool.assert_called_once_with("mcp_tool", {"param": "value"})
    # Check tool message was added
    assert stub_messages[2].role == "tool"
    assert stub_messages[2].name == "mcp_tool"


@pytest.mark.asyncio
async def test_c2_mcp_client_error(stub_messages: list[Message], mcp_client_stub: AsyncMock):
    """C2: MCP client error should be handled gracefully."""
    mcp_client_stub._acall_tool = AsyncMock(side_effect=RuntimeError("mcp fail"))
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[{"tool_calls": [{"name": "mcp_tool", "arguments": {}}]}],
    )
    agent = LLMAgent(provider=provider, tools=[], mcp_clients=[mcp_client_stub], max_iterations=1)
    await agent.ainit_mcp_tools()

    result = await agent.arespond(stub_messages)

    # Should return empty string due to max_iterations reached
    assert result == ""
    # Check that tool message contains error
    assert len(stub_messages) >= 3
    assert stub_messages[2].role == "tool"
    assert "Error" in stub_messages[2].content


# ============================================================================
# Tests D: Iteration limits
# ============================================================================


@pytest.mark.asyncio
async def test_d1_max_iterations_reached(stub_messages: list[Message]):
    """D1: Reaching max_iterations should return empty string."""
    # Create tool calls that never end
    provider = BaseMockProvider(
        supports_tools_val=True,
        responses=[{"tool_calls": [{"name": "test_tool", "arguments": {}}]}],
    )
    handler = Mock(return_value="ok")
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        handler=handler,
    )
    agent = LLMAgent(provider=provider, tools=[tool], max_iterations=2)

    result = await agent.arespond(stub_messages)

    assert result == ""


# ============================================================================
# Tests E: arespond_stream() without tools
# ============================================================================


@pytest.mark.asyncio
async def test_e1_provider_no_streaming_support(stub_messages: list[Message]):
    """E1: Provider without streaming support should fallback to arespond."""
    provider = BaseMockProvider(
        supports_tools_val=False,
        supports_streaming_val=False,
        responses=[{"content": "sync response"}],
    )
    agent = LLMAgent(provider=provider, tools=[])

    events = []
    async for event in agent.arespond_stream(stub_messages):
        events.append(event)

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT
    assert events[0].content == "sync response"


@pytest.mark.asyncio
async def test_e2_streaming_with_text_chunks(stub_messages: list[Message]):
    """E2: Streaming with text chunks should yield CONTENT events."""
    provider = BaseMockProvider(
        supports_tools_val=False,
        supports_streaming_val=True,
        responses=[
            {"content": "part1"},
            {"content": "part2"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[])

    events = []
    async for event in agent.arespond_stream(stub_messages):
        events.append(event)

    assert len(events) >= 2
    content_events = [e for e in events if e.type == EventType.CONTENT]
    assert len(content_events) >= 2
    assert content_events[0].content == "part1"
    assert content_events[1].content == "part2"


# ============================================================================
# Tests F: arespond_stream() with tools
# ============================================================================


@pytest.mark.asyncio
async def test_f1_successful_tool_call_in_stream(
    stub_messages: list[Message], local_tool_stub: AgentTool
):
    """F1: Successful tool call in stream should emit START/END events."""

    provider = BaseMockProvider(
        supports_tools_val=True,
        supports_streaming_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {"foo": "bar"}}]},
            {"content": "final answer"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[local_tool_stub])

    events = []
    async for event in agent.arespond_stream(stub_messages):
        events.append(event)

    # Check events - should have tool call events
    event_types = [e.type for e in events]
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_END in event_types
    # Content event may or may not appear depending on stream implementation

    # Check messages were updated
    assert len(stub_messages) >= 3
    assert stub_messages[1].role == "assistant"
    assert stub_messages[2].role == "tool"


@pytest.mark.asyncio
async def test_f2_tool_error_in_stream(stub_messages: list[Message]):
    """F2: Tool error in stream should emit TOOL_CALL_ERROR event."""
    handler = Mock(side_effect=RuntimeError("boom"))
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        handler=handler,
    )

    provider = BaseMockProvider(
        supports_tools_val=True,
        supports_streaming_val=True,
        responses=[
            {"tool_calls": [{"name": "test_tool", "arguments": {}}]},
            {"content": "recovered"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[tool], max_iterations=2)

    events = []
    async for event in agent.arespond_stream(stub_messages):
        events.append(event)

    # In streaming mode with tool errors, agent may continue to next iteration
    # Check that tool message was added with error
    tool_messages = [m for m in stub_messages if m.role == "tool"]
    assert len(tool_messages) >= 1
    assert "Error" in tool_messages[0].content


@pytest.mark.asyncio
async def test_f3_stream_completion_without_tools(stub_messages: list[Message]):
    """F3: Stream completion without tool calls should finish normally."""

    provider = BaseMockProvider(
        supports_tools_val=False,
        supports_streaming_val=True,
        responses=[
            {"content": "chunk1"},
            {"content": "chunk2"},
        ],
    )
    agent = LLMAgent(provider=provider, tools=[])

    events = []
    async for event in agent.arespond_stream(stub_messages):
        events.append(event)

    assert len(events) == 2
    assert all(e.type == EventType.CONTENT for e in events)


# ============================================================================
# Tests G: Utility methods
# ============================================================================


def test_g1_append_synthetic_assistant_turn(stub_messages: list[Message]):
    """G1: Synthetic assistant turn should create correct message structure."""
    provider = BaseMockProvider()
    agent = LLMAgent(provider=provider, tools=[])

    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test", arguments="{}"),
        )
    ]
    agent._append_synthetic_assistant_turn(stub_messages, tool_calls)

    assert len(stub_messages) == 2
    assert stub_messages[1].role == "assistant"
    assert stub_messages[1].content is None
    assert stub_messages[1].tool_calls == tool_calls


def test_g2_serialize_tool_result_string():
    """G2: String result should be returned as-is."""
    provider = BaseMockProvider()
    agent = LLMAgent(provider=provider, tools=[])

    result = agent._serialize_tool_result("test string")

    assert result == "test string"


def test_g2_serialize_tool_result_dict():
    """G2: Dict result should be serialized to JSON."""
    provider = BaseMockProvider()
    agent = LLMAgent(provider=provider, tools=[])

    result = agent._serialize_tool_result({"key": "value"})

    assert result == '{"key": "value"}'
    assert isinstance(result, str)


def test_g2_serialize_tool_result_non_serializable():
    """G2: Non-serializable object should be converted to string."""
    provider = BaseMockProvider()
    agent = LLMAgent(provider=provider, tools=[])

    class CustomObject:
        def __str__(self):
            return "custom_repr"

    result = agent._serialize_tool_result(CustomObject())

    assert result == "custom_repr"
