import json

import pytest
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch
from donkit.llm import (
    FunctionCall,
    GenerateRequest,
    GenerateResponse,
    LLMModelAbstract,
    Message,
    ModelCapability,
    StreamChunk,
    ToolCall,
)

# Ensure src/ is on sys.path for tests without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("RAGOPS_API_URL", "http://localhost:8080")


@pytest.fixture(autouse=True)
def disable_loguru():
    """Disable loguru logging during tests to prevent pytest capture issues."""
    import loguru

    # Remove all handlers and disable logging
    loguru.logger.remove()
    # Add a null handler to prevent any logging
    loguru.logger.add(lambda x: None, level="CRITICAL", enqueue=False)
    yield
    # Clean up after test
    loguru.logger.remove()


@pytest.fixture(autouse=True)
def use_plain_ui():
    """Force PlainUI for all tests to avoid prompt_toolkit file handle issues."""
    from donkit_ragops.ui import reset_ui, set_ui_adapter, UIAdapter

    reset_ui()
    set_ui_adapter(UIAdapter.PLAIN)
    yield
    reset_ui()


# ============================================================================
# Reusable Mock Classes
# ============================================================================


class BaseMockProvider(LLMModelAbstract):
    """Base mock LLM provider for testing with configurable responses.

    Can be extended or configured for different test scenarios.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        supports_tools_val: bool = True,
        supports_streaming_val: bool = False,
        model_name_val: str = "mock-model",
    ) -> None:
        """Initialize mock provider.

        Args:
            responses: List of response configurations. Each dict can contain:
                - 'content': Text response
                - 'tool_calls': List of tool calls to return
            supports_tools_val: Whether provider supports tools (default: True)
            supports_streaming_val: Whether provider supports streaming (default: False)
            model_name_val: Model name to return (default: "mock-model")
        """

        self.call_count = 0
        self.stream_call_count = 0
        self.messages_history: list[list[Message]] = []
        self.responses = responses or [{"content": "Default response"}]
        self.supports_tools_val = supports_tools_val
        self.supports_streaming_val = supports_streaming_val
        self._model_name = model_name_val

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set the model name/identifier."""
        self._model_name = value

    @property
    def capabilities(self) -> ModelCapability:
        """Return the capabilities supported by this model."""
        caps = ModelCapability.TEXT_GENERATION
        if self.supports_tools_val:
            caps |= ModelCapability.TOOL_CALLING
        if self.supports_streaming_val:
            caps |= ModelCapability.STREAMING
        return caps

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a response for the given request."""
        self.call_count += 1
        self.messages_history.append(request.messages.copy())

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
                    ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        type="function",
                        function=FunctionCall(
                            name=tc["name"],
                            arguments=json.dumps(args),
                        ),
                    )
                )

        return GenerateResponse(
            content=response_config.get("content"),
            tool_calls=tool_calls,
        )

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response for the given request."""
        self.stream_call_count += 1
        self.messages_history.append(request.messages.copy())

        # For streaming, yield each response as a separate chunk
        for response_config in self.responses:
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
                        ToolCall(
                            id=tc.get("id", f"call_{i}"),
                            type="function",
                            function=FunctionCall(
                                name=tc["name"],
                                arguments=json.dumps(args),
                            ),
                        )
                    )

            yield StreamChunk(
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
        with patch("donkit_ragops.mcp.client.Client") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client_instance

            with patch("donkit_ragops.mcp.client.StdioTransport"):
                yield mock_client_class, mock_client_instance

    return _create_mock


@pytest.fixture
def cli_mocks():
    """Pre-patched CLI dependencies for testing.

    Returns:
        Tuple of (mock_setup, mock_select, mock_repl)
    """
    with patch("donkit_ragops.cli.run_setup_if_needed") as mock_setup, patch(
        "donkit_ragops.cli.select_model_at_startup"
    ) as mock_select, patch("donkit_ragops.cli._run_local_mode") as mock_repl:
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
