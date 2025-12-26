# Reusable Test Components

This document describes reusable components available in `conftest.py` for use across all test files.

## Mock Classes

### BaseMockProvider

Base mock LLM provider with configurable responses.

**Usage:**
```python
from conftest import BaseMockProvider

# Simple text response
provider = BaseMockProvider(responses=[
    {"content": "Hello!"}
])

# With tool calls
provider = BaseMockProvider(responses=[
    {
        "tool_calls": [
            {"name": "tool_a", "arguments": {"x": 5}}
        ]
    },
    {"content": "Final answer"}
])
```

**Features:**
- Configurable response sequence
- Automatic cycling through responses
- Tracks call count and message history
- Supports both text and tool call responses

### BaseMockMCPClient

Base mock MCP client for testing agent + MCP integration.

**Usage:**
```python
from conftest import BaseMockMCPClient

client = BaseMockMCPClient(
    name="test_server",
    tools={
        "add": {
            "description": "Add two numbers",
            "parameters": {"type": "object"},
            "handler": lambda args: str(args["a"] + args["b"])
        }
    }
)

# Use in tests
tools = await client._alist_tools()
result = await client._acall_tool("add", {"a": 1, "b": 2})
```

## Fixtures

### mocked_mcp_client

Pre-configured MCP client mock with common setup. Eliminates repetitive patching code.

**Before:**
```python
with patch("donkit_ragops.mcp.client.Client") as mock_client_class:
    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)
    mock_client_class.return_value = mock_client_instance
    
    with patch("donkit_ragops.mcp.client.StdioTransport"):
        # test code
```

**After:**
```python
def test_something(mocked_mcp_client):
    with mocked_mcp_client() as (mock_class, mock_instance):
        mock_instance.list_tools = AsyncMock(return_value=[...])
        # test code
```

**Saves:** ~8-10 lines per test

### cli_mocks

Pre-patched CLI dependencies for testing.

**Before:**
```python
@patch("donkit_ragops.cli.run_setup_if_needed")
@patch("donkit_ragops.cli.select_model_at_startup")
@patch("donkit_ragops.cli._astart_repl")
def test_cli_something(mock_repl, mock_select, mock_setup):
    mock_setup.return_value = True
    mock_select.return_value = ("openai", "gpt-4")
    mock_repl.return_value = None
    # test code
```

**After:**
```python
def test_cli_something(cli_mocks):
    mock_setup, mock_select, mock_repl = cli_mocks
    # Default values already set, customize if needed
    # test code
```

**Saves:** ~6-8 lines per test

## Helper Functions

### create_mock_tool()

Factory for creating mock tool configurations.

**Usage:**
```python
from conftest import create_mock_tool

tool = create_mock_tool(
    name="add",
    handler=lambda args: str(args["a"] + args["b"]),
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    }
)
```

### assert_tool_has_valid_metadata()

Validates that a tool has all required metadata fields.

**Usage:**
```python
from conftest import assert_tool_has_valid_metadata

tools = await client._alist_tools()
for tool in tools:
    assert_tool_has_valid_metadata(tool)
```

**Replaces:**
```python
# Instead of this repetitive code:
assert "name" in tool
assert "description" in tool
assert "parameters" in tool
assert isinstance(tool["name"], str)
# ...
```

### assert_tool_schema_is_valid()

Validates JSON Schema structure.

**Usage:**
```python
from conftest import assert_tool_schema_is_valid

schema = tool["parameters"]
assert_tool_schema_is_valid(schema)
```
