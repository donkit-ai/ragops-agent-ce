"""Comprehensive tests for all MCP Servers using FastMCP Client.

Tests all 7 MCP servers:
2. chunker_server - Document chunking
3. compose_manager_server - Docker compose management
4. planner_server - RAG config planning
5. rag_query_server - RAG query execution
6. read_engine_server - Document reading/parsing
7. vectorstore_loader_server - Vector store loading

Includes:
- Tool listing and metadata validation
- Tool inventory verification (all expected tools present)
- Error handling for invalid tools and arguments
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from donkit_ragops.mcp.servers.chunker_server import server as chunker_server
from donkit_ragops.mcp.servers.compose_manager_server import server as compose_server
from donkit_ragops.mcp.servers.planner_server import server as planner_server
from donkit_ragops.mcp.servers.rag_query_server import server as query_server
from donkit_ragops.mcp.servers.read_engine_server import server as read_engine_server
from donkit_ragops.mcp.servers.vectorstore_loader_server import server as vectorstore_server

# ============================================================================
# Server Configuration
# ============================================================================

SERVERS = {
    "chunker": chunker_server,
    "compose": compose_server,
    "planner": planner_server,
    "query": query_server,
    "read_engine": read_engine_server,
    "vectorstore": vectorstore_server,
}

EXPECTED_TOOLS = {
    "chunker": {
        "chunk_documents",
    },
    "compose": {
        "get_logs",
        "init_project_compose",
        "list_available_services",
        "list_containers",
        "service_status",
        "start_service",
        "stop_container",
        "stop_service",
    },
    "planner": {
        "rag_config_plan",
    },
    "query": {
        "search_documents",
        "get_rag_prompt",
    },
    "read_engine": {
        "process_documents",
    },
    "vectorstore": {
        "vectorstore_load",
    },
}


# ============================================================================
# Parametrized Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def server_client(request) -> Client[FastMCPTransport]:
    """Create a FastMCP client for the requested server."""
    server_name = request.param
    server = SERVERS[server_name]
    async with Client(transport=server) as client:
        yield client


# ============================================================================
# Parametrized Tests - Basic Functionality
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_client",
    ["chunker", "compose", "planner", "query", "read_engine", "vectorstore"],
    indirect=True,
)
async def test_server_list_tools(server_client: Client[FastMCPTransport]) -> None:
    """Test that server lists tools correctly."""
    tools = await server_client.list_tools()
    assert len(tools) > 0
    tool_names = [t.name for t in tools]
    assert len(tool_names) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_client",
    ["chunker", "compose", "planner", "query", "read_engine", "vectorstore"],
    indirect=True,
)
async def test_server_tools_have_descriptions(
    server_client: Client[FastMCPTransport],
) -> None:
    """Test that all server tools have descriptions."""
    tools = await server_client.list_tools()
    for tool in tools:
        assert tool.description, f"Tool {tool.name} missing description"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_client",
    ["chunker", "compose", "planner", "query", "read_engine", "vectorstore"],
    indirect=True,
)
async def test_server_tools_have_schemas(server_client: Client[FastMCPTransport]) -> None:
    """Test that all server tools have input schemas."""
    tools = await server_client.list_tools()
    for tool in tools:
        assert tool.inputSchema, f"Tool {tool.name} missing inputSchema"
        assert "type" in tool.inputSchema, f"Tool {tool.name} schema missing 'type'"


# ============================================================================
# Parametrized Tests - Tool Inventory
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_client,server_name",
    [
        ("chunker", "chunker"),
        ("compose", "compose"),
        ("planner", "planner"),
        ("query", "query"),
        ("read_engine", "read_engine"),
        ("vectorstore", "vectorstore"),
    ],
    indirect=["server_client"],
)
async def test_server_inventory(
    server_client: Client[FastMCPTransport],
    server_name: str,
) -> None:
    """Test that server has all expected tools."""
    tools = await server_client.list_tools()
    tool_names = {t.name for t in tools}
    expected = EXPECTED_TOOLS[server_name]
    assert tool_names == expected, f"Tool mismatch for {server_name}"


# ============================================================================
# Individual Server Fixtures (for specific tests)
# ============================================================================


@pytest_asyncio.fixture
async def planner_client() -> Client[FastMCPTransport]:
    """Create a FastMCP client for planner server."""
    async with Client(transport=planner_server) as client:
        yield client


@pytest_asyncio.fixture
async def query_client() -> Client[FastMCPTransport]:
    """Create a FastMCP client for query server."""
    async with Client(transport=query_server) as client:
        yield client


# ============================================================================
# Server-Specific Tests
# ============================================================================


@pytest.mark.asyncio
async def test_planner_server_rag_config_plan_missing_args(
    planner_client: Client[FastMCPTransport],
) -> None:
    """Test rag_config_plan validation."""
    with pytest.raises(Exception):
        # Missing required arguments - should raise an error
        await planner_client.call_tool("rag_config_plan", {})


@pytest.mark.asyncio
async def test_query_server_search_documents_missing_query(
    query_client: Client[FastMCPTransport],
) -> None:
    """Test search_documents validation."""
    with pytest.raises(Exception):
        await query_client.call_tool("search_documents", {})


@pytest.mark.asyncio
async def test_query_server_get_rag_prompt_missing_query(
    query_client: Client[FastMCPTransport],
) -> None:
    """Test get_rag_prompt validation."""
    with pytest.raises(Exception):
        await query_client.call_tool("get_rag_prompt", {})


# ============================================================================
# Cross-Server Tests
# ============================================================================


@pytest_asyncio.fixture
async def all_clients():
    """Create clients for all servers."""
    async with (
        Client(transport=chunker_server) as chunker,
        Client(transport=compose_server) as compose,
        Client(transport=planner_server) as planner,
        Client(transport=query_server) as query,
        Client(transport=read_engine_server) as read_engine,
        Client(transport=vectorstore_server) as vectorstore,
    ):
        clients = {
            "chunker": chunker,
            "compose": compose,
            "planner": planner,
            "query": query,
            "read_engine": read_engine,
            "vectorstore": vectorstore,
        }
        yield clients


@pytest.mark.asyncio
async def test_all_servers_respond_to_list_tools(all_clients) -> None:
    """Test that all servers respond to list_tools."""
    for server_name, client in all_clients.items():
        tools = await client.list_tools()
        assert len(tools) > 0, f"{server_name} returned no tools"


@pytest.mark.asyncio
async def test_all_servers_nonexistent_tool_error(all_clients) -> None:
    """Test that all servers raise errors for nonexistent tools."""
    for server_name, client in all_clients.items():
        with pytest.raises(Exception):
            await client.call_tool("nonexistent_tool_xyz", {})
