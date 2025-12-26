"""WebSocket client for enterprise mode.

Connects to API Gateway via WebSocket to interact with the server-side agent.
History is saved automatically on the server side.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable
from uuid import UUID

from loguru import logger


class WebSocketClient:
    """WebSocket client for enterprise mode chat.

    Connects to API Gateway WebSocket endpoint to send messages
    and receive responses from the server-side agent.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        project_id: UUID | str,
        on_agent_thinking: Callable[[], None] | None = None,
        on_tool_call: Callable[[str], None] | None = None,
        on_reconnect: Callable[[], None] | None = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            base_url: API Gateway base URL (e.g., https://api.donkit.ai)
            api_token: Authentication token
            project_id: Project ID to connect to
            on_agent_thinking: Callback when agent starts thinking
            on_tool_call: Callback when agent makes a tool call
            on_reconnect: Callback when reconnecting
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.project_id = str(project_id)
        self.on_agent_thinking = on_agent_thinking or (lambda: None)
        self.on_tool_call = on_tool_call or (lambda _: None)
        self.on_reconnect = on_reconnect or (lambda: None)

        self._ws = None
        self._connected = False
        self._response_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL."""
        base = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/agent/ws?project_id={self.project_id}"

    async def run(self, message_handler: Callable[[dict], None] | None = None) -> None:
        """
        Run the WebSocket connection loop.

        This is the main loop that handles connection, reconnection,
        and message receiving. Should be run as a background task.

        Args:
            message_handler: Optional callback for incoming messages
        """
        from websockets.asyncio.client import connect

        headers = {"X-API-Token": self.api_token}

        while True:
            try:
                async with connect(self.ws_url, additional_headers=headers) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info(f"Connected to WebSocket at {self.ws_url}")

                    async for message in ws:
                        try:
                            event = json.loads(message)
                        except json.JSONDecodeError:
                            event = {"text": message}

                        event_type = event.get("type")

                        if event_type == "agent_thinking":
                            self.on_agent_thinking()
                        elif event_type == "chat_message":
                            data = event.get("data", {})
                            role = data.get("role", "")

                            if role == "tool":
                                tool_content = data.get("content", "")
                                self.on_tool_call(tool_content)

                            # Put response in queue for awaiting caller
                            await self._response_queue.put(data)

                            # Also call handler if provided
                            if message_handler:
                                message_handler(data)
                        else:
                            logger.debug(f"Received unknown event type: {event_type}")

            except asyncio.CancelledError:
                logger.info("WebSocket task cancelled")
                break
            except Exception as e:
                logger.warning(f"WebSocket disconnected: {e}")
                self._connected = False
                self._ws = None
                self.on_reconnect()
                await asyncio.sleep(1)  # Wait before reconnecting

    async def send_message(
        self,
        text: str,
        attached_files: list[str] | None = None,
        file_analysis: dict | None = None,
    ) -> AsyncIterator[dict]:
        """
        Send a message and yield responses.

        Args:
            text: User message text
            attached_files: List of S3 paths to attached files
            file_analysis: File analysis metadata

        Yields:
            Response messages from the agent
        """
        if not self._ws or not self._connected:
            raise RuntimeError("WebSocket not connected")

        # Clear any pending responses
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send message
        payload = {
            "type": "chat_message",
            "data": {
                "text": text,
                "attached_files": attached_files or [],
                "file_analysis": file_analysis or {},
            },
        }

        await self._ws.send(json.dumps(payload, ensure_ascii=False))
        logger.debug(f"Sent message: {text[:50]}...")

        # Wait for responses
        # The server sends multiple messages: tool calls, final response, etc.
        # We yield each one until we get the final assistant message
        while True:
            try:
                response = await asyncio.wait_for(
                    self._response_queue.get(),
                    timeout=300.0,  # 5 minute timeout
                )

                if response is None:
                    break

                yield response

                # Stop after receiving final assistant message (not a tool call)
                role = response.get("role", "")
                content = response.get("content", "")
                if role == "assistant" and content and not response.get("tool_calls"):
                    break

            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for response")
                break

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("WebSocket closed")
