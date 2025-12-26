"""Enterprise REPL implementation.

Handles enterprise mode REPL with WebSocket connection to cloud server.
Uses UI abstraction layer for all output (PromptToolkitUI adapter).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from donkit_ragops import texts
from donkit_ragops.prints import RAGOPS_LOGO_ART, RAGOPS_LOGO_TEXT
from donkit_ragops.repl.base import BaseREPL, ReplContext
from donkit_ragops.ui import UIAdapter, get_ui, set_ui_adapter
from donkit_ragops.ui.styles import StyleName

if TYPE_CHECKING:
    from donkit.ragops_api_gateway_client.client import RagopsAPIGatewayClient


def _render_markdown_simple(text: str) -> str:
    """Render markdown to plain text.

    Simple markdown rendering without Rich markup.
    """
    result = text

    # Bold **text** -> just text
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result, flags=re.DOTALL)

    # Italic *text* -> just text
    result = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"\1", result)

    # Code `text` -> text
    result = re.sub(r"`([^`]+)`", r"\1", result)

    # Headers # text -> text
    result = re.sub(r"^#+\s+(.+)$", r"\1", result, flags=re.MULTILINE)

    # List items - text -> • text
    result = re.sub(r"^(\s*)[-*]\s+", r"\1• ", result, flags=re.MULTILINE)

    return result


class EnterpriseWsHandler:
    """WebSocket handler for enterprise mode CLI."""

    def __init__(self, project_id: str) -> None:
        """Initialize handler.

        Args:
            project_id: Project ID for the session
        """
        self.project_id = project_id
        self.ws: Any = None
        self._running = True
        self._thinking = False
        self._spinner: Any = None
        self._response_complete = asyncio.Event()
        self._response_complete.set()  # Initially complete (no pending response)

    async def handle_ws_connection(self, ws_connection: Any) -> None:
        """Handle WebSocket connection.

        Args:
            ws_connection: WebSocket connection object
        """
        self.ws = ws_connection
        ui = get_ui()
        ui.print("Connected to agent", StyleName.SUCCESS)

        async for message in self.ws:
            if not self._running:
                break

            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                event = {"text": message}

            event_type = event.get("type")

            if event_type == "agent_thinking":
                self._thinking = True
                # Spinner is already started in send_message(), no need to start again
                # Just ensure it's running (in case of reconnection scenarios)
                if not self._spinner:
                    self._spinner = ui.create_spinner(texts.THINKING_MESSAGE_PLAIN)
                    self._spinner.start()

            elif event_type == "chat_message":
                # Stop spinner
                if self._spinner:
                    self._spinner.stop()
                    self._spinner = None
                self._thinking = False

                data = event.get("data", {})
                role = data.get("role", "")
                content = data.get("content", "").strip()

                if role == "tool":
                    ui.print(f"Tool: {content[:100]}...", StyleName.DIM)
                elif content:
                    # Parse JSON content if needed
                    if content.startswith("{"):
                        try:
                            parsed = json.loads(content)
                            content = parsed.get("text", content)
                        except json.JSONDecodeError:
                            pass

                    # Print agent response
                    ui.newline()
                    ui.print("Agent:", StyleName.AGENT_PREFIX)
                    rendered = _render_markdown_simple(content)
                    ui.print(rendered)
                    ui.newline()

                    # Mark response as complete
                    self._response_complete.set()

    async def send_message(self, text: str, attached_files: list[str], file_analysis: dict) -> None:
        """Send a message via WebSocket.

        Args:
            text: Message text
            attached_files: List of attached file paths
            file_analysis: File analysis data
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected")

        # Clear response event BEFORE sending to avoid race condition
        # This ensures wait_for_response() will actually wait
        self._response_complete.clear()

        # Start spinner immediately when sending (not waiting for agent_thinking)
        ui = get_ui()
        self._thinking = True
        self._spinner = ui.create_spinner(texts.THINKING_MESSAGE_PLAIN)
        self._spinner.start()

        payload = {
            "type": "chat_message",
            "data": {
                "text": text,
                "attached_files": attached_files,
                "file_analysis": file_analysis,
            },
        }
        await self.ws.send(json.dumps(payload, ensure_ascii=False))

    async def wait_for_response(self, timeout: float = 300.0) -> bool:
        """Wait for the agent response to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if response completed, False if timed out
        """
        try:
            await asyncio.wait_for(self._response_complete.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def stop(self) -> None:
        """Stop the handler."""
        self._running = False
        if self._spinner:
            self._spinner.stop()
            self._spinner = None


class EnterpriseREPL(BaseREPL):
    """REPL for enterprise mode with WebSocket connection.

    Handles file uploads and cloud-based agent execution.
    """

    def __init__(
        self,
        context: ReplContext,
        api_client: RagopsAPIGatewayClient,
        token: str,
        enterprise_settings: Any,
    ) -> None:
        """Initialize EnterpriseREPL.

        Args:
            context: REPL context
            api_client: API client for cloud operations
            token: Auth token
            enterprise_settings: Enterprise configuration
        """
        super().__init__(context)
        self.api_client = api_client
        self.token = token
        self.enterprise_settings = enterprise_settings
        self.project_id: str | None = None
        self.ws_handler: EnterpriseWsHandler | None = None
        self.ws_task: asyncio.Task | None = None
        self.attached_files: list[str] = []

        # Set UI adapter to prompt_toolkit for enterprise mode
        set_ui_adapter(UIAdapter.RICH)

    async def initialize(self) -> None:
        """Initialize REPL resources."""
        ui = get_ui()

        ui.print(RAGOPS_LOGO_TEXT)
        ui.print(RAGOPS_LOGO_ART)

        # Create project
        ui.print("Creating project...", StyleName.DIM)
        try:
            async with self.api_client:
                project = await self.api_client.create_project()
                self.project_id = str(project.id)
                ui.print(f"Project: {self.project_id[:8]}...", StyleName.SUCCESS)
        except Exception as e:
            ui.print_error(f"Error creating project: {e}")
            self.stop()
            return

        # Create WebSocket handler
        self.ws_handler = EnterpriseWsHandler(self.project_id)
        self.api_client.register_ws_handler(self.ws_handler)

        # Connect to WebSocket
        ui.print("Connecting to server...", StyleName.DIM)
        self.ws_task = asyncio.create_task(self._connect_ws())

        # Wait for connection
        for _ in range(50):  # 5 seconds max
            if self.ws_handler.ws:
                break
            await asyncio.sleep(0.1)

        if not self.ws_handler.ws:
            ui.print_error("Connection failed")
            self.stop()

    async def _connect_ws(self) -> None:
        """Connect to WebSocket with auto-reconnect."""
        from donkit.ragops_api_gateway_client.errors import (
            RagopsAPIGatewayConnectionError,
            RagopsAPIGatewayMaxAttemptsExceededError,
        )

        ui = get_ui()

        while self._running and self.ws_handler:
            try:
                await self.api_client.connect_ws(self.project_id)
            except RagopsAPIGatewayMaxAttemptsExceededError:
                ui.print_error("Failed to connect after max attempts.")
                return
            except RagopsAPIGatewayConnectionError:
                ui.print_warning("Connection error. Reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                if self._running:
                    ui.print_warning(f"Disconnected: {e}. Reconnecting...")
                    await asyncio.sleep(1)
                else:
                    break

    async def run(self) -> None:
        """Run the REPL main loop."""
        await self.initialize()

        if not self._running:
            return

        ui = get_ui()
        ui.newline()
        ui.print("Enterprise Mode - Type your message or :quit to exit", StyleName.BOLD)
        ui.print("Use @/path/to/file to attach files", StyleName.DIM)
        ui.newline()

        while self._running:
            try:
                user_input = await asyncio.to_thread(ui.text_input)
            except KeyboardInterrupt:
                ui.print_warning("Press Ctrl+C again or type :quit to exit")
                continue

            if not user_input:
                continue

            should_continue = await self.handle_input(user_input)
            if not should_continue:
                break

        await self.cleanup()

    async def handle_input(self, user_input: str) -> bool:
        """Handle user input.

        Args:
            user_input: Raw user input

        Returns:
            False to exit, True to continue
        """
        ui = get_ui()

        if user_input in {":q", ":quit", ":exit", "exit", "quit"}:
            ui.print("Goodbye!", StyleName.DIM)
            return False

        if user_input == ":help":
            ui.newline()
            ui.print("Commands:", StyleName.BOLD)
            ui.print("  :quit, :exit - Exit the application")
            ui.print("  :clear - Clear attached files")
            ui.print("  @/path/to/file - Attach a file")
            ui.newline()
            return True

        if user_input == ":clear":
            self.attached_files.clear()
            ui.print("Cleared attached files", StyleName.DIM)
            return True

        # Check for file attachments
        if not user_input.startswith(":"):
            file_path = Path(user_input)
            if file_path.exists():
                return await self._handle_file_upload(file_path)

        # Regular message
        await self.handle_message(user_input)
        return True

    async def _handle_file_upload(self, file_path: Path) -> bool:
        """Handle file upload.

        Args:
            file_path: Path to file or directory

        Returns:
            True to continue REPL
        """
        from donkit_ragops.enterprise.analyzer import FileAnalyzer
        from donkit_ragops.enterprise.upload import FileUploader

        ui = get_ui()

        files_to_upload: list[str] = []
        if file_path.is_file():
            files_to_upload.append(str(file_path))
        elif file_path.is_dir():
            files_in_dir = list(file_path.rglob("*"))
            files_to_upload.extend(str(f) for f in files_in_dir if f.is_file())

        if not files_to_upload:
            return True

        file_names = [Path(f).name for f in files_to_upload]
        ui.newline()
        ui.print(f"Uploading: {', '.join(file_names)}", StyleName.INFO)

        file_analyzer = FileAnalyzer()
        file_uploader = FileUploader(self.api_client)

        s3_paths: list[str] = []
        file_analysis: dict = {}

        # Use spinner for analysis
        with ui.create_spinner("Analyzing files...") as spinner:
            try:
                file_analysis = await file_analyzer.analyze_files(
                    [Path(f) for f in files_to_upload]
                )
                spinner.update("Analysis complete")
            except Exception as e:
                ui.print_warning(f"Analysis failed: {e}")

        # Use progress bar for upload
        with ui.create_progress(len(files_to_upload), "Uploading to cloud...") as progress:
            async with self.api_client:
                for i, file_path_str in enumerate(files_to_upload):
                    s3_path = await file_uploader.upload_single_file(file_path_str, self.project_id)
                    if s3_path:
                        s3_paths.append(s3_path)
                    progress.update(i + 1)

            file_uploader.reset()

        if not s3_paths:
            ui.print_error("Upload failed")
            return True

        ui.print_success(f"Uploaded {len(s3_paths)} file(s)")

        # Send auto message about files
        if len(file_names) == 1:
            auto_message = f"[Attached file: {file_names[0]}]"
        else:
            auto_message = f"[Attached {len(file_names)} files: {', '.join(file_names)}]"

        ui.newline()
        ui.print(f"You: {auto_message}", StyleName.INFO)
        ui.newline()

        try:
            await self.ws_handler.send_message(auto_message, s3_paths, file_analysis)
            # Wait for response to complete before showing next input
            await self.ws_handler.wait_for_response()
        except (KeyboardInterrupt, asyncio.CancelledError):
            ui.print_warning("Response interrupted")
            self.ws_handler._response_complete.set()
        except Exception as e:
            ui.print_error(str(e))
            logger.error(f"Error sending file message: {e}", exc_info=True)

        return True

    async def handle_message(self, message: str) -> None:
        """Handle a chat message.

        Args:
            message: User's chat message
        """
        ui = get_ui()

        if not self.ws_handler:
            ui.print_error("Not connected")
            return

        try:
            await self.ws_handler.send_message(message, [], {})
            # Wait for response to complete before showing next input
            await self.ws_handler.wait_for_response()
        except (KeyboardInterrupt, asyncio.CancelledError):
            ui.print_warning("Response interrupted")
            self.ws_handler._response_complete.set()  # Unblock on interrupt
        except Exception as e:
            ui.print_error(str(e))
            logger.error(f"Error in enterprise REPL: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Clean up REPL resources."""
        self._running = False

        if self.ws_handler:
            self.ws_handler.stop()

        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass

        ui = get_ui()
        ui.print("Disconnected", StyleName.DIM)
