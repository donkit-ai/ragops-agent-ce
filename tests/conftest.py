import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from ragops_agent_ce.llm.types import LLMResponse
from ragops_agent_ce.llm.types import Message
from ragops_agent_ce.llm.types import ToolCall
from ragops_agent_ce.llm.types import ToolFunctionCall

# Ensure src/ is on sys.path for tests without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("RAGOPS_API_URL", "http://localhost:8080")


# ============================================================================
# Reusable Mock Classes
# ============================================================================


class BaseMockProvider:
    """Base mock LLM provider for testing with configurable responses.

    Can be extended or configured for different test scenarios.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        supports_tools_val: bool = True,
        supports_streaming_val: bool = False,
    ) -> None:
        """Initialize mock provider.

        Args:
            responses: List of response configurations. Each dict can contain:
                - 'content': Text response
                - 'tool_calls': List of tool calls to return
            supports_tools_val: Whether provider supports tools (default: True)
            supports_streaming_val: Whether provider supports streaming (default: False)
        """

        self.call_count = 0
        self.stream_call_count = 0
        self.messages_history: list[list[Message]] = []
        self.responses = responses or [{"content": "Default response"}]
        self.supports_tools_val = supports_tools_val
        self.supports_streaming_val = supports_streaming_val
        self._LLMResponse = LLMResponse
        self._ToolCall = ToolCall
        self._ToolFunctionCall = ToolFunctionCall

    def supports_tools(self) -> bool:
        """Support tools."""
        return self.supports_tools_val

    def supports_streaming(self) -> bool:
        """Support streaming."""
        return self.supports_streaming_val

    def generate(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        model: str | None = None,
    ) -> Any:
        """Generate response based on configured responses."""
        self.call_count += 1
        self.messages_history.append(messages.copy())

        # Get response config for current call (cycle if needed)
        response_idx = (self.call_count - 1) % len(self.responses)
        response_config = self.responses[response_idx]

        # Build tool calls if provided
        tool_calls = None
        if "tool_calls" in response_config:
            tool_calls = []
            for i, tc in enumerate(response_config["tool_calls"]):
                # Parse arguments if they're a string
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                tool_calls.append(
                    self._ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        type="function",
                        function=self._ToolFunctionCall(
                            name=tc["name"],
                            arguments=args,
                        ),
                    )
                )

        return self._LLMResponse(
            content=response_config.get("content"),
            tool_calls=tool_calls,
        )

    def generate_stream(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        model: str | None = None,
    ) -> Any:
        """Stream responses based on configured responses."""
        self.stream_call_count += 1
        self.messages_history.append(messages.copy())

        # Get response config for current call (cycle if needed)
        response_idx = (self.stream_call_count - 1) % len(self.responses)
        response_config = self.responses[response_idx]

        # Build tool calls if provided
        tool_calls = None
        if "tool_calls" in response_config:
            tool_calls = []
            for i, tc in enumerate(response_config["tool_calls"]):
                # Parse arguments if they're a string
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                tool_calls.append(
                    self._ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        type="function",
                        function=self._ToolFunctionCall(
                            name=tc["name"],
                            arguments=args,
                        ),
                    )
                )

        yield self._LLMResponse(
            content=response_config.get("content"),
            tool_calls=tool_calls,
        )


class BaseMockMCPClient:
    """Base mock MCP client for testing.

    Provides minimal functionality for testing agent + MCP integration.
    """

    def __init__(self, name: str, tools: dict[str, Any]) -> None:
        """Initialize mock MCP client.

        Args:
            name: Client name
            tools: Dict mapping tool names to tool configs
                Each tool config can contain:
                - 'description': Tool description
                - 'parameters': Tool parameters schema
                - 'handler': Callable that handles the tool call
        """
        self.name = name
        self.tools = tools
        self.call_count = 0

    async def _alist_tools(self) -> list[dict]:
        """List available tools."""
        return [
            {
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "parameters": tool_info.get("parameters", {}),
            }
            for tool_name, tool_info in self.tools.items()
        ]

    async def _acall_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool."""
        self.call_count += 1

        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")

        tool_info = self.tools[name]
        handler = tool_info.get("handler")

        if handler:
            return handler(arguments)

        return json.dumps({"result": "success"})


# ============================================================================
# Reusable Fixtures
# ============================================================================


@pytest.fixture
def mocked_mcp_client():
    """Pre-configured MCP client mock with common setup.

    Yields a context manager that provides mock_client_class and mock_client_instance.

    Usage:
        with mocked_mcp_client() as (mock_class, mock_instance):
            # Configure mock_instance.list_tools, etc.
            # Your test code here
    """

    @contextmanager
    def _create_mock():
        with patch("ragops_agent_ce.mcp.client.Client") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client_instance

            with patch("ragops_agent_ce.mcp.client.StdioTransport"):
                yield mock_client_class, mock_client_instance

    return _create_mock


@pytest.fixture
def cli_mocks():
    """Pre-patched CLI dependencies for testing.

    Returns:
        Tuple of (mock_setup, mock_select, mock_repl)
    """
    with patch("ragops_agent_ce.cli.run_setup_if_needed") as mock_setup, patch(
        "ragops_agent_ce.cli.select_model_at_startup"
    ) as mock_select, patch("ragops_agent_ce.cli._astart_repl") as mock_repl:
        # Default return values
        mock_setup.return_value = True
        mock_select.return_value = ("openai", "gpt-4")
        mock_repl.return_value = None

        yield mock_setup, mock_select, mock_repl


# ============================================================================
# Helper Functions
# ============================================================================


def create_mock_tool(
    name: str,
    handler: Any,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Factory for creating mock tool configurations.

    Args:
        name: Tool name
        handler: Callable that handles tool execution
        description: Tool description
        parameters: Tool parameters schema

    Returns:
        Tool configuration dict
    """
    return {
        "name": name,
        "description": description or f"Mock tool: {name}",
        "parameters": parameters
        or {
            "type": "object",
            "properties": {},
        },
        "handler": handler,
    }


def assert_tool_has_valid_metadata(tool: dict[str, Any]) -> None:
    """Assert that tool has all required metadata fields.

    Args:
        tool: Tool dict to validate
    """
    assert "name" in tool, "Tool missing 'name'"
    assert "description" in tool, "Tool missing 'description'"
    assert "parameters" in tool, "Tool missing 'parameters'"
    assert isinstance(tool["name"], str), "Tool name must be string"
    assert isinstance(tool["description"], str), "Tool description must be string"
    assert isinstance(tool["parameters"], dict), "Tool parameters must be dict"


def assert_tool_schema_is_valid(schema: dict[str, Any]) -> None:
    """Assert that tool schema is valid JSON Schema.

    Args:
        schema: JSON Schema to validate
    """
    assert "type" in schema, "Schema missing 'type'"
    assert schema["type"] == "object", "Schema type must be 'object'"

    if "properties" in schema:
        assert isinstance(schema["properties"], dict), "Schema properties must be dict"

    if "required" in schema:
        assert isinstance(schema["required"], list), "Schema required must be list"
