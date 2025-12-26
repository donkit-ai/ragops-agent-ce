"""Cloud history synchronization for enterprise mode.

Handles saving and loading message history to/from cloud storage.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class HistorySyncError(Exception):
    """Error during history synchronization."""

    pass


class CloudHistory:
    """Manages cloud history synchronization."""

    def __init__(self, api_client):
        """
        Initialize cloud history manager.

        Args:
            api_client: RagopsAPIGatewayClient or similar with add_message/get_history methods
        """
        self.api_client = api_client

    async def save_message(
        self,
        project_id: str,
        role: str,
        content: str | None = None,
        attached_files: list[str] | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Save a message to cloud history.

        Args:
            project_id: Project ID
            role: Message role (user, assistant, tool)
            content: Message content
            attached_files: List of S3 paths to attached files
            tool_name: Tool name (for tool responses)
            tool_call_id: Tool call ID (for tool responses)
            tool_calls: Tool calls (for assistant messages with tool use)

        Returns:
            Created message info with id and created_at
        """
        payload = {
            "project_id": project_id,
            "message": {
                "role": role,
                "content": content,
                "attached_files": attached_files,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "tool_calls": tool_calls,
            },
        }

        try:
            async with self.api_client:
                result = await self.api_client.add_message(payload)
            logger.debug(f"Saved {role} message to cloud: {result.get('id', 'unknown')}")
            return result
        except Exception as e:
            logger.error(f"Failed to save message to cloud: {e}")
            raise HistorySyncError(f"Failed to save message: {e}") from e

    async def save_user_message(
        self,
        project_id: str,
        content: str,
        attached_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Save a user message to cloud history.

        Args:
            project_id: Project ID
            content: Message content
            attached_files: List of S3 paths to attached files

        Returns:
            Created message info
        """
        return await self.save_message(
            project_id=project_id,
            role="user",
            content=content,
            attached_files=attached_files,
        )

    async def save_assistant_message(
        self,
        project_id: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Save an assistant message to cloud history.

        Args:
            project_id: Project ID
            content: Message content
            tool_calls: Tool calls made by assistant

        Returns:
            Created message info
        """
        return await self.save_message(
            project_id=project_id,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    async def save_tool_message(
        self,
        project_id: str,
        content: str,
        tool_name: str,
        tool_call_id: str,
    ) -> dict[str, Any]:
        """
        Save a tool response message to cloud history.

        Args:
            project_id: Project ID
            content: Tool response content
            tool_name: Name of the tool
            tool_call_id: ID of the tool call this responds to

        Returns:
            Created message info
        """
        return await self.save_message(
            project_id=project_id,
            role="tool",
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    async def load_history(self, project_id: str) -> list[dict[str, Any]]:
        """
        Load message history from cloud.

        Args:
            project_id: Project ID

        Returns:
            List of messages in project chat history
        """
        try:
            async with self.api_client:
                history = await self.api_client.get_history(project_id)
            logger.debug(f"Loaded {len(history)} messages from cloud history")
            return history
        except Exception as e:
            logger.error(f"Failed to load history from cloud: {e}")
            raise HistorySyncError(f"Failed to load history: {e}") from e


async def save_messages_to_cloud(
    api_client,
    project_id: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convenience function to save multiple messages to cloud.

    Args:
        api_client: API client with add_message method
        project_id: Project ID
        messages: List of messages with role, content, etc.

    Returns:
        List of created message infos
    """
    history = CloudHistory(api_client)
    results = []

    for msg in messages:
        result = await history.save_message(
            project_id=project_id,
            role=msg.get("role", "assistant"),
            content=msg.get("content"),
            attached_files=msg.get("attached_files"),
            tool_name=msg.get("tool_name"),
            tool_call_id=msg.get("tool_call_id"),
            tool_calls=msg.get("tool_calls"),
        )
        results.append(result)

    return results
