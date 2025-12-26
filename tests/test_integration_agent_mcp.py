"""Integration tests for Agent + MCP — agent with MCP servers.

These tests verify that the agent correctly:
1. Discovers MCP tools at startup
2. Calls MCP tools and processes results
3. Works with multiple MCP clients
4. Handles MCP server errors gracefully
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest
from donkit_ragops.agent.agent import LLMAgent
from donkit_ragops.agent.local_tools.tools import AgentTool
from donkit.llm import LLMModelAbstract, StreamChunk
from donkit.llm import GenerateRequest
from donkit.llm import GenerateResponse
from donkit.llm import Message
from donkit.llm import ModelCapability
from donkit.llm import ToolCall
from donkit.llm import FunctionCall

from .conftest import BaseMockMCPClient
from .conftest import BaseMockProvider

# ============================================================================
# Test-specific configurations
# ============================================================================


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_provider() -> BaseMockProvider:
    """Create a mock provider with MCP tool call scenario."""
    return BaseMockProvider(
        responses=[
            {"tool_calls": [{"name": "mcp_tool_a", "arguments": {"x": 10}}]},
            {"content": "MCP tool returned: 100"},
        ]
    )


@pytest.fixture
def mock_mcp_client() -> BaseMockMCPClient:
    """Create a mock MCP client."""

    def mcp_tool_a_handler(args: dict[str, Any]) -> str:
        """Handler for mcp_tool_a."""
        x = args.get("x", 0)
        result = x * x
        return json.dumps({"result": result})

    return BaseMockMCPClient(
        name="test_mcp",
        tools={
            "mcp_tool_a": {
                "description": "MCP Tool A",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
                "handler": mcp_tool_a_handler,
            },
            "mcp_tool_b": {
                "description": "MCP Tool B",
                "parameters": {"type": "object", "properties": {}},
                "handler": lambda args: json.dumps({"result": "tool_b_result"}),
            },
        },
    )


# ============================================================================
# Tests: MCP Tool Discovery
# ============================================================================


@pytest.mark.asyncio
async def test_agent_discovers_mcp_tools(
    mock_provider: BaseMockProvider,
    mock_mcp_client: BaseMockMCPClient,
) -> None:
    """Test agent discovers MCP tools at startup."""
    agent = LLMAgent(provider=mock_provider, mcp_clients=[mock_mcp_client])

    # Initialize MCP tools
    await agent.ainit_mcp_tools()

    # Should have discovered MCP tools
    assert len(agent.mcp_tools) == 2
    assert "mcp_tool_a" in agent.mcp_tools
    assert "mcp_tool_b" in agent.mcp_tools


@pytest.mark.asyncio
async def test_agent_mcp_tools_in_specs(
    mock_provider: BaseMockProvider,
    mock_mcp_client: BaseMockMCPClient,
) -> None:
    """Test that MCP tools appear in tool specs."""
    agent = LLMAgent(provider=mock_provider, mcp_clients=[mock_mcp_client])

    await agent.ainit_mcp_tools()

    specs = agent._tool_specs()

    # Should have MCP tools in specs
    spec_names = [s.function.name for s in specs]
    assert "mcp_tool_a" in spec_names
    assert "mcp_tool_b" in spec_names


# ============================================================================
# Tests: MCP Tool Calling
# ============================================================================


@pytest.mark.asyncio
async def test_agent_calls_mcp_tool(
    mock_provider: BaseMockProvider,
    mock_mcp_client: BaseMockMCPClient,
) -> None:
    """Test agent calls MCP tool and gets result."""
    agent = LLMAgent(provider=mock_provider, mcp_clients=[mock_mcp_client])

    await agent.ainit_mcp_tools()

    messages = [
        Message(role="user", content="Use MCP tool A with x=10"),
    ]

    result = await agent.arespond(messages)

    # Should have called MCP tool
    assert mock_mcp_client.call_count == 1
    # Should have final response
    assert "100" in result or "result" in result.lower()


@pytest.mark.asyncio
async def test_agent_mcp_tool_result_in_messages(
    mock_provider: BaseMockProvider,
    mock_mcp_client: BaseMockMCPClient,
) -> None:
    """Test that MCP tool results are added to messages."""
    agent = LLMAgent(provider=mock_provider, mcp_clients=[mock_mcp_client])

    await agent.ainit_mcp_tools()

    messages = [
        Message(role="user", content="Use MCP tool"),
    ]

    await agent.arespond(messages)

    # Should have tool result in messages
    assert any(m.role == "tool" for m in messages)
    tool_message = next(m for m in messages if m.role == "tool")
    assert "100" in tool_message.content or "result" in tool_message.content.lower()


# ============================================================================
# Tests: Multiple MCP Clients
# ============================================================================


@pytest.mark.asyncio
async def test_agent_with_multiple_mcp_clients(
    mock_provider: BaseMockProvider,
) -> None:
    """Test agent with multiple MCP clients."""
    client1 = BaseMockMCPClient(
        name="mcp1",
        tools={
            "tool_from_mcp1": {
                "description": "Tool from MCP 1",
                "parameters": {},
                "handler": lambda args: json.dumps({"source": "mcp1"}),
            }
        },
    )

    client2 = BaseMockMCPClient(
        name="mcp2",
        tools={
            "tool_from_mcp2": {
                "description": "Tool from MCP 2",
                "parameters": {},
                "handler": lambda args: json.dumps({"source": "mcp2"}),
            }
        },
    )

    agent = LLMAgent(provider=mock_provider, mcp_clients=[client1, client2])

    await agent.ainit_mcp_tools()

    # Should have tools from both clients
    assert "tool_from_mcp1" in agent.mcp_tools
    assert "tool_from_mcp2" in agent.mcp_tools
    assert len(agent.mcp_tools) == 2


# ============================================================================
# Tests: MCP Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_agent_handles_mcp_tool_error() -> None:
    """Test agent handles MCP tool errors."""

    class ErrorMCPClient:
        """MCP client that raises errors."""

        async def _alist_tools(self) -> list[dict]:
            """List tools."""
            return [
                {
                    "name": "error_tool",
                    "description": "Tool that errors",
                    "parameters": {},
                }
            ]

        async def _acall_tool(self, name: str, arguments: dict[str, Any]) -> str:
            """Call tool - always fails."""
            raise RuntimeError("MCP tool execution failed")

    class ErrorProvider(LLMModelAbstract):
        """Provider that calls error tool."""

        def __init__(self):
            self._model_name = "error-mock"

        @property
        def model_name(self) -> str:
            return self._model_name

        @model_name.setter
        def model_name(self, value: str) -> None:
            self._model_name = value

        @property
        def capabilities(self) -> ModelCapability:
            return ModelCapability.TEXT_GENERATION | ModelCapability.TOOL_CALLING

        async def generate(self, request: GenerateRequest) -> GenerateResponse:
            return GenerateResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="error_tool",
                            arguments="{}",
                        ),
                    )
                ],
            )

        async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
            raise NotImplementedError()

    agent = LLMAgent(provider=ErrorProvider(), mcp_clients=[ErrorMCPClient()], max_iterations=1)

    await agent.ainit_mcp_tools()

    messages = [Message(role="user", content="Call error tool")]

    # Should return empty string due to max_iterations reached
    result = await agent.arespond(messages)
    assert result == ""


# ============================================================================
# Tests: MCP Tool with Complex Results
# ============================================================================


@pytest.mark.asyncio
async def test_agent_processes_complex_mcp_result(
    mock_provider: BaseMockProvider,
) -> None:
    """Test agent processes complex MCP tool results."""

    class ComplexMCPClient:
        """MCP client returning complex results."""

        async def _alist_tools(self) -> list[dict]:
            """List tools."""
            return [
                {
                    "name": "complex_tool",
                    "description": "Tool returning complex data",
                    "parameters": {},
                }
            ]

        async def _acall_tool(self, name: str, arguments: dict[str, Any]) -> str:
            """Call tool."""
            result = {
                "status": "success",
                "data": {
                    "items": [
                        {"id": 1, "name": "item1"},
                        {"id": 2, "name": "item2"},
                    ],
                    "total": 2,
                },
                "metadata": {"timestamp": "2025-11-17T14:54:00Z"},
            }
            return json.dumps(result)

    class ComplexProvider(LLMModelAbstract):
        def __init__(self) -> None:
            self.call_count = 0
            self._model_name = "complex-mock"

        @property
        def model_name(self) -> str:
            return self._model_name

        @model_name.setter
        def model_name(self, value: str) -> None:
            self._model_name = value

        @property
        def capabilities(self) -> ModelCapability:
            return ModelCapability.TEXT_GENERATION | ModelCapability.TOOL_CALLING

        async def generate(self, request: GenerateRequest) -> GenerateResponse:
            self.call_count += 1

            if self.call_count == 1:
                return GenerateResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=FunctionCall(
                                name="complex_tool",
                                arguments="{}",
                            ),
                        )
                    ],
                )

            # Verify tool result is in messages
            for msg in request.messages:
                if msg.role == "tool":
                    data = json.loads(msg.content)
                    assert data["status"] == "success"
                    assert len(data["data"]["items"]) == 2

            return GenerateResponse(content="Processed complex result")

        def generate_stream(self, messages, tools=None, model=None):
            raise NotImplementedError()

    agent = LLMAgent(
        provider=ComplexProvider(),
        mcp_clients=[ComplexMCPClient()],
    )

    await agent.ainit_mcp_tools()

    messages = [Message(role="user", content="Get complex data")]

    result = await agent.arespond(messages)

    assert result  # Should have result


# ============================================================================
# Tests: Mixed Local and MCP Tools
# ============================================================================


@pytest.mark.asyncio
async def test_agent_with_local_and_mcp_tools(
    mock_provider: BaseMockProvider,
    mock_mcp_client: BaseMockMCPClient,
) -> None:
    """Test agent with both local and MCP tools."""

    def local_tool_handler(args: dict[str, Any]) -> str:
        """Local tool handler."""
        return json.dumps({"source": "local"})

    local_tools = [
        AgentTool(
            name="local_tool",
            description="A local tool",
            handler=local_tool_handler,
            parameters={"type": "object", "properties": {}},
        ),
    ]

    agent = LLMAgent(
        provider=mock_provider,
        tools=local_tools,
        mcp_clients=[mock_mcp_client],
    )

    await agent.ainit_mcp_tools()

    specs = agent._tool_specs()
    spec_names = [s.function.name for s in specs]

    # Should have both local and MCP tools
    assert "local_tool" in spec_names
    assert "mcp_tool_a" in spec_names
    assert "mcp_tool_b" in spec_names
