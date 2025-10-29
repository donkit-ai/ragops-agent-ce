from __future__ import annotations

import json

from loguru import logger

from ..llm.base import LLMProvider
from ..llm.types import Message, ToolSpec
from ..mcp.client import MCPClient
from .project_tools import (
    tool_create_project,
    tool_get_project,
    tool_get_rag_config,
    tool_list_projects,
    tool_save_rag_config,
)
from .tools import AgentTool, tool_db_get, tool_grep, tool_list_directory, tool_time_now


def default_tools() -> list[AgentTool]:
    return [
        tool_time_now(),
        tool_db_get(),
        tool_list_directory(),
        tool_grep(),
        tool_create_project(),
        tool_get_project(),
        tool_list_projects(),
        tool_save_rag_config(),
        tool_get_rag_config(),
    ]


class LLMAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[AgentTool] | None = None,
        mcp_clients: list[MCPClient] | None = None,
        max_iterations: int = 30,
    ) -> None:
        self.provider = provider
        self.local_tools = tools or default_tools()
        self.mcp_clients = mcp_clients or []
        self.mcp_tools: dict[str, tuple[dict, MCPClient]] = {}

        for client in self.mcp_clients:
            try:
                discovered = client.list_tools()
                for t in discovered:
                    tool_name = t["name"]
                    # t["parameters"] = _clean_schema_for_vertex(t["parameters"])
                    self.mcp_tools[tool_name] = (t, client)
            except Exception:
                logger.error(
                    f"Failed to list tools from MCP client {client.command}", exc_info=True
                )
                pass
        self.max_iterations = max_iterations

    def _tool_specs(self) -> list[ToolSpec]:
        specs = [t.to_tool_spec() for t in self.local_tools]
        for tool_info, _ in self.mcp_tools.values():
            specs.append(
                ToolSpec(
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
        return bool(self.provider.supports_tools() and resp.tool_calls)

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

    def _execute_tool_call(self, tc, args: dict) -> str:
        """Execute either a local or MCP tool and return a serialized string result.

        Raises on execution error, matching previous behavior.
        """
        try:
            local_tool, mcp_tool_info = self._find_tool(tc.function.name)
            if not local_tool and not mcp_tool_info:
                logger.warning(f"Tool not found: {tc.function.name}")
                return ""

            if local_tool:
                result = local_tool.handler(args)
                logger.debug(f"Local tool {tc.function.name} result: {str(result)[:200]}...")
            elif mcp_tool_info:
                tool_meta, client = mcp_tool_info
                result = client.call_tool(tool_meta["name"], args)
                logger.debug(f"MCP tool {tc.function.name} result: {str(result)[:200]}...")
            else:
                result = f"Error: Tool '{tc.function.name}' not found or MCP client not configured."
                logger.error(result)
        except Exception as e:
            result = f"MALFORMED_FUNCTION_CALL: Error executing tool '{tc.function.name}': {e}"
            logger.error(f"Tool execution error: {e}", exc_info=True)
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

    def _handle_tool_calls(self, messages: list[Message], tool_calls) -> None:
        """Full tool call handling: synthetic assistant turn, execute, and append tool messages."""
        logger.debug(f"Processing {len(tool_calls)} tool calls")
        # 1) synthetic assistant turns
        self._append_synthetic_assistant_turn(messages, tool_calls)
        # 2) execute and append responses
        for tc in tool_calls:
            args = self._parse_tool_args(tc)
            result_str = self._execute_tool_call(tc, args)
            messages.append(
                Message(
                    role="tool",
                    name=tc.function.name,
                    tool_call_id=tc.id,
                    content=result_str,
                )
            )

    def chat(self, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return self.respond(messages, model=model)

    def respond(self, messages: list[Message], *, model: str | None = None) -> str:
        """Perform a single assistant turn given an existing message history.

        This method mutates the provided messages list by appending tool results as needed.
        Returns the assistant content.
        """
        tools = self._tool_specs() if self.provider.supports_tools() else None

        for _ in range(self.max_iterations):
            resp = self.provider.generate(messages, tools=tools, model=model)

            # Handle tool calls if requested
            if self._should_execute_tools(resp):
                self._handle_tool_calls(messages, resp.tool_calls)
                # continue loop to give tool results back to the model
                continue

            # Otherwise return the content from the model
            if not resp.content:
                retry_resp = self.provider.generate(messages, model=model)
                return retry_resp.content or ""
            return resp.content

        return ""
