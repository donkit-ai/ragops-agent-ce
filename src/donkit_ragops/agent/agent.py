from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum, auto

from donkit.llm import GenerateRequest, LLMModelAbstract, Message, ModelCapability, Tool
from loguru import logger

from donkit_ragops.agent.local_tools.checklist_tools import (
    tool_create_checklist,
    tool_get_checklist,
    tool_update_checklist_item,
)
from donkit_ragops.agent.local_tools.project_tools import (
    tool_add_loaded_files,
    tool_create_project,
    tool_delete_project,
    tool_get_project,
    tool_get_rag_config,
    tool_list_loaded_files,
    tool_list_projects,
    tool_save_rag_config,
)
from donkit_ragops.agent.local_tools.tools import (
    AgentTool,
    tool_db_get,
    tool_grep,
    tool_interactive_user_choice,
    tool_interactive_user_confirm,
    tool_list_directory,
    tool_quick_start_rag_config,
    tool_read_file,
    tool_time_now,
    tool_update_rag_config_field,
)
from donkit_ragops.mcp.client import MCPClient


class EventType(StrEnum):
    CONTENT = auto()
    TOOL_CALL_START = auto()
    TOOL_CALL_END = auto()
    TOOL_CALL_ERROR = auto()


@dataclass
class StreamEvent:
    """Event yielded during streaming response."""

    type: EventType
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    error: str | None = None


def default_tools() -> list[AgentTool]:
    return [
        tool_time_now(),
        tool_db_get(),
        tool_list_directory(),
        tool_read_file(),
        tool_grep(),
        tool_interactive_user_choice(),
        tool_interactive_user_confirm(),
        tool_quick_start_rag_config(),
        tool_update_rag_config_field(),
        tool_create_project(),
        tool_get_project(),
        tool_list_projects(),
        tool_delete_project(),
        tool_save_rag_config(),
        tool_get_rag_config(),
        tool_add_loaded_files(),
        tool_list_loaded_files(),
        # Checklist management tools
        tool_create_checklist(),
        tool_get_checklist(),
        tool_update_checklist_item(),
    ]


class LLMAgent:
    def __init__(
        self,
        provider: LLMModelAbstract,
        tools: list[AgentTool] | None = None,
        mcp_clients: list[MCPClient] | None = None,
        max_iterations: int = 50,
    ) -> None:
        self.provider = provider
        self.local_tools = tools or default_tools()
        self.mcp_clients = mcp_clients or []
        self.mcp_tools: dict[str, tuple[dict, MCPClient]] = {}
        self.max_iterations = max_iterations

    async def ainit_mcp_tools(self) -> None:
        """Initialize MCP tools asynchronously. Call this after creating the agent."""
        for client in self.mcp_clients:
            try:
                discovered = await client._alist_tools()
                for t in discovered:
                    tool_name = t["name"]
                    # t["parameters"] = _clean_schema_for_vertex(t["parameters"])
                    self.mcp_tools[tool_name] = (t, client)
            except Exception:
                logger.error(
                    f"Failed to list tools from MCP client {client.command}", exc_info=True
                )
                pass

    def _tool_specs(self) -> list[Tool]:
        specs = [t.to_tool_spec() for t in self.local_tools]
        for tool_info, _ in self.mcp_tools.values():
            specs.append(
                Tool(
                    **{
                        "function": {
                            "name": tool_info["name"],
                            "description": tool_info["description"],
                            "parameters": tool_info["parameters"],
                        }
                    }
                )
            )
        return specs

    def _find_tool(self, name: str) -> tuple[AgentTool | None, tuple[dict, MCPClient] | None]:
        for t in self.local_tools:
            if t.name == name:
                return t, None
        if name in self.mcp_tools:
            return None, self.mcp_tools[name]
        return None, None

    # --- Internal helpers to keep respond() small and readable ---
    def _should_execute_tools(self, resp) -> bool:
        """Whether the provider response requires tool execution."""
        return bool(
            self.provider.supports_capability(ModelCapability.TOOL_CALLING) and resp.tool_calls
        )

    def _append_synthetic_assistant_turn(self, messages: list[Message], tool_calls) -> None:
        """Append a single assistant message with tool_calls."""
        messages.append(
            Message(
                role="assistant",
                content=None,  # No text content when calling tools
                tool_calls=tool_calls,
            )
        )

    def _parse_tool_args(self, tc) -> dict:
        """Parse tool arguments into a dict, tolerating stringified JSON or None."""
        try:
            raw = tc.function.arguments
            if isinstance(raw, dict):
                return raw
            return json.loads(raw or "{}")
        except Exception as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            return {}

    async def _aexecute_tool_call(self, tc, args: dict) -> str:
        """Execute either a local or MCP tool and return a serialized string result.

        Raises on execution error, matching previous behavior.
        """
        try:
            local_tool, mcp_tool_info = self._find_tool(tc.function.name)
            if not local_tool and not mcp_tool_info:
                logger.warning(f"Tool not found: {tc.function.name}")
                return ""

            if local_tool:
                logger.debug(f"Executing local tool {tc.function.name} with args: {args}")
                result = local_tool.handler(args)
                logger.debug(f"Local tool {tc.function.name} result: {str(result)[:200]}...")
            elif mcp_tool_info:
                logger.debug(f"Executing MCP tool {tc.function.name} with args: {args}")
                tool_meta, client = mcp_tool_info
                result = await client._acall_tool(tool_meta["name"], args)
                logger.debug(f"MCP tool {tc.function.name} result: {str(result)[:200]}...")
            else:
                result = f"Error: Tool '{tc.function.name}' not found or MCP client not configured."
                logger.error(result)

        except KeyboardInterrupt:
            logger.warning(f"Tool {tc.function.name} execution cancelled by user")
            # Don't raise - return cancellation message instead
            return "Tool execution cancelled by user (Ctrl+C)"
        except asyncio.CancelledError:
            logger.warning(f"Tool {tc.function.name} execution cancelled")
            # Don't raise - return cancellation message instead
            return "Tool execution cancelled by user (Ctrl+C)"
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            # Return error message as tool result
            return f"Error: {str(e)}"

        return self._serialize_tool_result(result)

    def _serialize_tool_result(self, result) -> str:
        """Ensure the tool result is a JSON string."""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to serialize tool result to JSON: {e}")
            return str(result)

    async def _ahandle_tool_calls(self, messages: list[Message], tool_calls) -> None:
        """Full tool call handling: synthetic assistant turn, execute, and append tool messages."""
        logger.debug(f"Processing {len(tool_calls)} tool calls")
        # 1) synthetic assistant turns
        self._append_synthetic_assistant_turn(messages, tool_calls)
        # 2) execute and append responses
        for tc in tool_calls:
            args = self._parse_tool_args(tc)
            result_str = await self._aexecute_tool_call(tc, args)
            messages.append(
                Message(
                    role="tool",
                    name=tc.function.name,
                    tool_call_id=tc.id,
                    content=result_str,
                )
            )

    async def achat(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> str:
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return await self.arespond(messages)

    async def achat_stream(
        self, *, prompt: str, system: str | None = None, model: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Chat with streaming output. Yields text chunks."""
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        async for chunk in self.arespond_stream(messages):
            yield chunk

    async def arespond(self, messages: list[Message]) -> str:
        """Perform a single assistant turn given an existing message history.

        This method mutates the provided messages list by appending tool results as needed.
        Returns the assistant content.
        """
        tools = (
            self._tool_specs()
            if self.provider.supports_capability(ModelCapability.TOOL_CALLING)
            else None
        )

        for _ in range(self.max_iterations):
            request = GenerateRequest(messages=messages, tools=tools)
            resp = await self.provider.generate(request)

            # Handle tool calls if requested
            if self._should_execute_tools(resp):
                await self._ahandle_tool_calls(messages, resp.tool_calls)
                # continue loop to give tool results back to the model
                continue

            # Otherwise return the content from the model
            if not resp.content:
                retry_request = GenerateRequest(messages=messages)
                retry_resp = await self.provider.generate(retry_request)
                return retry_resp.content or ""
            return resp.content

        return ""

    async def arespond_stream(self, messages: list[Message]) -> AsyncIterator[StreamEvent]:
        """Perform a single assistant turn with streaming output.

        This method mutates the provided messages list by appending tool results as needed.
        Yields StreamEvent objects for content chunks and tool calls.

        Returns:
            AsyncIterator that yields StreamEvent objects.
        """
        tools = (
            self._tool_specs()
            if self.provider.supports_capability(ModelCapability.TOOL_CALLING)
            else None
        )

        for _ in range(self.max_iterations):
            request = GenerateRequest(messages=messages, tools=tools)
            async for chunk in self.provider.generate_stream(request):  # noqa
                # Yield text chunks as they arrive
                if chunk.content:
                    yield StreamEvent(type=EventType.CONTENT, content=chunk.content)

                # Handle tool calls immediately when they arrive
                if chunk.tool_calls and self.provider.supports_capability(
                    ModelCapability.TOOL_CALLING
                ):
                    # Append synthetic assistant turn
                    self._append_synthetic_assistant_turn(messages, chunk.tool_calls)

                    # Execute each tool and yield events
                    for tc in chunk.tool_calls:
                        args = self._parse_tool_args(tc)

                        # Yield tool call start event
                        yield StreamEvent(
                            type=EventType.TOOL_CALL_START,
                            tool_name=tc.function.name,
                            tool_args=args,
                        )

                        try:
                            # Execute tool
                            result_str = await self._aexecute_tool_call(tc, args)
                            # Add the tool result to messages
                            messages.append(
                                Message(
                                    role="tool",
                                    name=tc.function.name,
                                    tool_call_id=tc.id,
                                    content=result_str,
                                )
                            )
                            # Yield tool call end event
                            yield StreamEvent(
                                type=EventType.TOOL_CALL_END, tool_name=tc.function.name
                            )
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Tool {tc.function.name} failed: {error_msg}")
                            # Add an error as the tool result
                            messages.append(
                                Message(
                                    role="tool",
                                    name=tc.function.name,
                                    tool_call_id=tc.id,
                                    content=f"Error: {error_msg}",
                                )
                            )
                            # Yield the tool call error event
                            yield StreamEvent(
                                type=EventType.TOOL_CALL_ERROR,
                                tool_name=tc.function.name,
                                error=error_msg,
                            )
                    # Break inner loop to start new iteration with tool results
                    break
            else:
                # Stream finished without tool calls - done
                return
            # Continue outer loop - send tool results back to model
        # Max iterations reached
        return
