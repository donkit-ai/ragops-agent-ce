from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, find_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from loguru import logger


def _load_env_for_mcp() -> dict[str, str | None]:
    """Load environment variables for MCP server.

    Combines:
    1. Current os.environ (so MCP server inherits parent env)
    2. Variables from .env files (with multiple search strategies for Windows compatibility)

    Returns dict with environment variables for the MCP server process.
    """
    # Start with current environment
    env = dict(os.environ)

    # Try to load from .env files in multiple locations
    env_loaded = False
    for fname in (".env.local", ".env"):
        # 1. Current working directory
        cwd_path = Path.cwd() / fname
        if cwd_path.exists():
            env.update(dotenv_values(cwd_path))
            env_loaded = True
            logger.debug(f"Loaded MCP env from {cwd_path}")
        if env_loaded:
            break
        # 2. Parent directories (walk up 3 levels)
        parent = Path.cwd()
        for _ in range(4):
            parent = parent.parent
            parent_env = parent / fname
            if parent_env.exists():
                env.update(dotenv_values(parent_env))
                env_loaded = True
                logger.debug(f"Loaded MCP env from {parent_env}")
                break
        # 3. Fallback to find_dotenv
        if not env_loaded:
            found = find_dotenv(filename=fname, usecwd=True)
            if found:
                env.update(dotenv_values(found))
                env_loaded = True
                logger.debug(f"Loaded MCP env from {found}")

    if not env_loaded:
        logger.debug("No .env file found for MCP server, using current environment only")
    return env


class MCPClient:
    """Client for connecting to an MCP server using FastMCP.

    Uses the new FastMCP Client API which handles connection lifecycle
    and protocol operations automatically.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        timeout: float = 999.0,
        progress_callback: Callable[[float, float | None, str | None], None] | None = None,
    ) -> None:
        """Initialize MCP client.

        Args:
            command: Command to run the MCP server (e.g., "python" or "uv")
            args: Command-line arguments including the script path
            timeout: Timeout for operations in seconds
            progress_callback: Optional callback for progress updates from MCP tools
        """
        self.command = command
        self.args = args or []
        self.timeout = timeout
        self.progress_callback = progress_callback
        # Load environment variables for the server
        self._env = _load_env_for_mcp()

    async def __progress_handler(
        self,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        """Handle progress updates from read_engine MCP server."""
        if self.progress_callback:
            self.progress_callback(progress, total, message)
        else:
            # Fallback to print if no callback provided
            if total is not None:
                percentage = (progress / total) * 100
                print(f"Progress: {percentage:.1f}% - {message or ''}")
            else:
                print(f"Progress: {progress} - {message or ''}")

    async def _alist_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        # Create StdioTransport with explicit command, args, and environment
        transport = StdioTransport(
            command=self.command,
            args=self.args,
            env=self._env,
        )
        client = Client(transport)
        try:
            async with client:
                tools_resp = await client.list_tools()
                tools = []
                for t in tools_resp:
                    # FastMCP returns Tool objects with name, description, and inputSchema (dict)
                    schema: dict[str, Any] = {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    }
                    # Access inputSchema attribute
                    # (note: lowercase 's' in input_schema or inputSchema)
                    raw_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                    if raw_schema and isinstance(raw_schema, dict):
                        try:
                            # FastMCP wraps Pydantic models in {"args": <model>}
                            # Unwrap by following $ref to get actual model schema
                            if "properties" in raw_schema:
                                if "args" in raw_schema["properties"] and "$defs" in raw_schema:
                                    # Get the ref target
                                    args_ref = raw_schema["properties"]["args"].get("$ref")
                                    if args_ref and args_ref.startswith("#/$defs/"):
                                        def_name = args_ref.split("/")[-1]
                                        if def_name in raw_schema["$defs"]:
                                            # Use the unwrapped model schema
                                            schema = raw_schema["$defs"][def_name].copy()
                                            # Preserve $defs for nested refs
                                            if "$defs" in raw_schema:
                                                schema["$defs"] = raw_schema["$defs"]
                                else:
                                    # No args wrapper, use as is
                                    schema = raw_schema
                        except Exception as e:
                            logger.warning(f"Failed to parse schema for tool {t.name}: {e}")
                    tools.append(
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": schema,
                        }
                    )
                return tools
        except asyncio.CancelledError:
            logger.warning("Tool listing was cancelled")
            raise
        finally:
            # Ensure transport cleanup even on interruption
            if hasattr(transport, "_process") and transport._process:
                try:
                    transport._process.terminate()
                    await asyncio.sleep(0.1)  # Give it time to terminate
                    if transport._process.poll() is None:
                        transport._process.kill()
                except Exception as e:
                    logger.debug(f"Error during transport cleanup: {e}")

    def list_tools(self) -> list[dict]:
        """Synchronously list available tools."""
        try:
            return asyncio.run(asyncio.wait_for(self._alist_tools(), timeout=self.timeout))
        except KeyboardInterrupt:
            logger.warning("Tool listing interrupted by user")
            # Don't re-raise - return empty list to allow agent to continue
            return []
        finally:
            # Ensure any remaining event loop cleanup happens
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.stop()
            except RuntimeError:
                # No event loop available, which is fine
                pass

    async def _acall_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        logger.debug(f"Calling tool {name} with arguments {arguments}")
        # Create StdioTransport with explicit command, args, and environment
        transport = StdioTransport(command=self.command, args=self.args, env=self._env)
        client = Client(transport, progress_handler=self.__progress_handler)
        try:
            async with client:
                # FastMCP wraps Pydantic models in {"args": <model>}, so wrap arguments
                wrapped_args = {"args": arguments} if arguments else None
                logger.debug(f"Wrapped arguments for {name}: {wrapped_args}")
                result = await client.call_tool(name, wrapped_args)
                # FastMCP returns ToolResult with content and optional data
                # Try to extract text content first
                if hasattr(result, "content") and result.content:
                    texts: list[str] = []
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            texts.append(content_item.text)
                    if texts:
                        return "\n".join(texts)
                # Fall back to structured data if available
                if hasattr(result, "data") and result.data is not None:
                    if isinstance(result.data, str):
                        return result.data
                    return json.dumps(result.data)
                # Last resort: stringify the whole result
                return str(result)
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.warning(f"Tool {name} execution was cancelled")
            raise
        except KeyboardInterrupt:
            logger.warning(f"Tool {name} execution interrupted by user")
            raise
        finally:
            # Ensure transport cleanup even on interruption
            if hasattr(transport, "_process") and transport._process:
                try:
                    transport._process.terminate()
                    try:
                        await asyncio.sleep(0.1)  # Give it time to terminate
                    except asyncio.CancelledError:
                        pass
                    if transport._process.poll() is None:
                        transport._process.kill()
                except Exception as e:
                    logger.debug(f"Error during transport cleanup: {e}")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Synchronously call a tool."""
        try:
            result = asyncio.run(
                asyncio.wait_for(self._acall_tool(name, arguments), timeout=self.timeout)
            )
            if not isinstance(result, str):
                return json.dumps(result)
            return result
        except KeyboardInterrupt:
            logger.warning(f"Tool {name} execution interrupted by user")
            raise
        finally:
            # Ensure any remaining event loop cleanup happens
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.stop()
            except RuntimeError:
                # No event loop available, which is fine
                pass
