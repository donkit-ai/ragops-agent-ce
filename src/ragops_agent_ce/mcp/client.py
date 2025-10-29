from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Thin client for connecting to an MCP server via stdio transport.

    A new stdio session is created per call (simple and robust). If you need
    long-lived sessions or high performance, we can refactor to keep a shared
    running session.
    """

    def __init__(self, command: str, args: list[str] | None = None, timeout: float = 60.0) -> None:
        self.command = command
        self.args = args or []
        self.timeout = timeout

    async def _alist_tools(self) -> list[dict]:
        server_params = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools = []
                for t in tools_resp.tools:
                    # t has: name, description, input_schema (JSON Schema)
                    schema: dict[str, Any] = {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    }
                    # Support both snake_case and camelCase per SDK/version
                    cand = None
                    if hasattr(t, "input_schema"):
                        cand = getattr(t, "input_schema")
                    elif hasattr(t, "inputSchema"):
                        cand = getattr(t, "inputSchema")
                    if cand is not None:
                        try:
                            # Pydantic model
                            if hasattr(cand, "model_dump"):
                                schema = cand.model_dump()  # type: ignore[attr-defined]
                            elif isinstance(cand, dict):
                                schema = cand
                        except Exception:
                            pass
                    tools.append(
                        {
                            "name": t.name,
                            "description": getattr(t, "description", "") or "",
                            "parameters": schema,
                        }
                    )
                return tools

    def list_tools(self) -> list[dict]:
        return asyncio.run(asyncio.wait_for(self._alist_tools(), timeout=self.timeout))

    async def _acall_tool(self, name: str, arguments: dict[str, Any]) -> str:
        logger.debug(f"Calling tool {name} with arguments {arguments}")
        server_params = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                parts = getattr(result, "content", [])
                texts: list[str] = []
                for p in parts:
                    pt = getattr(p, "type", None)
                    if pt == "text" and hasattr(p, "text"):
                        texts.append(p.text)
                if texts:
                    return "\n".join(texts)
                try:
                    return json.dumps(parts[0]) if parts else ""
                except Exception:
                    return ""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = asyncio.run(
            asyncio.wait_for(self._acall_tool(name, arguments), timeout=self.timeout)
        )
        if not isinstance(result, str):
            return json.dumps(result)
        return result
