"""Cloud MCP Client for enterprise mode.

Uses native FastMCP Client with BearerAuth for connecting to the
Donkit Cloud MCP Server (API Gateway).
"""

from __future__ import annotations

from typing import Any

from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from loguru import logger


class CloudMCPClient:
    """MCP client for connecting to cloud server with authorization.

    This is a wrapper around fastmcp.Client that provides the same interface
    as the local MCPClient, allowing seamless integration with LLMAgent.

    Features:
    - Native FastMCP client with all features (SSE, Streamable HTTP)
    - Built-in authorization via BearerAuth
    - Automatic reconnection
    - Support for all MCP features (resources, prompts, sampling)
    """

    def __init__(self, url: str, token: str):
        """Initialize CloudMCPClient.

        Args:
            url: Cloud MCP server URL (e.g., https://api.donkit.ai/mcp)
            token: API token for authentication (Bearer token, without prefix)
        """
        self.url = url
        self.token = token
        self._client: Client | None = None
        self._is_connected = False

    async def __aenter__(self) -> CloudMCPClient:
        """Enter async context and connect to server."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context and disconnect."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to the cloud MCP server."""
        if self._is_connected:
            return

        logger.debug(f"Connecting to Cloud MCP at {self.url}")
        self._client = Client(
            self.url,
            auth=BearerAuth(token=self.token),
        )
        await self._client.__aenter__()
        self._is_connected = True
        logger.debug("Connected to Cloud MCP server")

    async def disconnect(self) -> None:
        """Disconnect from the cloud MCP server."""
        if self._client and self._is_connected:
            await self._client.__aexit__(None, None, None)
            self._is_connected = False
            logger.debug("Disconnected from Cloud MCP server")

    async def _alist_tools(self) -> list[dict]:
        """Get list of available tools from cloud MCP.

        Returns:
            List of tool definitions in format expected by LLMAgent:
            [{"name": str, "description": str, "parameters": dict}, ...]
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        tools = await self._client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {},
            }
            for tool in tools
        ]

    async def _acall_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on the cloud MCP server.

        Args:
            name: Tool name to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        logger.debug(f"Calling cloud tool: {name}")
        result = await self._client.call_tool(name, arguments)
        return result

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected

    # Alias for compatibility with existing MCPClient interface
    @property
    def command(self) -> str:
        """Return URL as command for logging compatibility."""
        return self.url
