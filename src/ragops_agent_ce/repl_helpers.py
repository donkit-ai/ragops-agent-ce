"""Helpers for rendering transcript and processing stream events in the REPL."""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field

from rich.markup import escape

from ragops_agent_ce import texts
from ragops_agent_ce.agent.agent import EventType
from ragops_agent_ce.agent.agent import StreamEvent
from ragops_agent_ce.checklist_manager import get_active_checklist_text
from ragops_agent_ce.checklist_manager import handle_checklist_tool_event
from ragops_agent_ce.display import ScreenRenderer
from ragops_agent_ce.llm.types import Message
from ragops_agent_ce.schemas.agent_schemas import AgentSettings


def format_timestamp() -> str:
    """Return timestamp string used in transcript lines."""
    return "[dim]" + time.strftime("[%H:%M]", time.localtime()) + "[/]"


@dataclass
class ReplRenderHelper:
    """Encapsulates transcript rendering and streaming utilities for the REPL."""

    transcript: list[str]
    renderer: ScreenRenderer
    agent_settings: AgentSettings
    session_started_at: float
    show_checklist: bool

    def render_current_screen(self, cl_text: str | None = None) -> None:
        """Render the combined transcript/checklist view."""
        if cl_text is None and self.show_checklist:
            cl_text = get_active_checklist_text(self.session_started_at)
        self.renderer.render_project(self.transcript, cl_text, agent_settings=self.agent_settings)

    def append_user_line(self, text: str) -> None:
        """Add a user input line to the transcript."""
        self.transcript.append(f"\n\n{format_timestamp()} {texts.USER_PREFIX} {escape(text)}")

    def append_agent_message(self, content: str) -> None:
        """Append an agent message (already formatted Markdown)."""
        self.transcript.append(f"{format_timestamp()} {texts.AGENT_PREFIX} {content}")

    def start_agent_placeholder(self) -> int:
        """Insert placeholder for streaming agent output."""
        self.transcript.append(f"\n{format_timestamp()} {texts.AGENT_PREFIX} ")
        return len(self.transcript) - 1

    def set_agent_line(self, index: int, display_content: str, temp_executing: str) -> None:
        """Replace placeholder line with current streaming content."""
        self.transcript[index] = (
            f"\n{format_timestamp()} {texts.AGENT_PREFIX} {display_content}{temp_executing}"
        )

    def append_error(self, message: str) -> None:
        self.transcript.append(f"{format_timestamp()} [bold red]Error:[/bold red] {message}")


@dataclass
class MCPEventHandler:
    """Handles MCP progress updates and stream events."""

    render_helper: ReplRenderHelper
    agent_settings: AgentSettings
    session_started_at: float
    show_checklist: bool
    progress_line_index: int | None = field(default=None, init=False)

    def progress_callback(self, progress: float, total: float | None, message: str | None) -> None:
        """Callback compatible with `MCPClient` progress updates."""
        if total is not None:
            percentage = (progress / total) * 100
            progress_text = texts.PROGRESS_PERCENTAGE.format(
                percentage=percentage, message=message or ""
            )
        else:
            progress_text = texts.PROGRESS_GENERIC.format(progress=progress, message=message or "")

        if self.progress_line_index is None:
            self.render_helper.transcript.append(progress_text)
            self.progress_line_index = len(self.render_helper.transcript) - 1
        else:
            self.render_helper.transcript[self.progress_line_index] = progress_text

        self.render_helper.render_current_screen()

    def clear_progress(self) -> None:
        if self.progress_line_index is not None:
            self.render_helper.transcript.pop(self.progress_line_index)
            self.progress_line_index = None

    @staticmethod
    def tool_executing_message(tool_name: str, tool_args: dict | None) -> str:
        args_str = ", ".join(tool_args.keys()) if tool_args else ""
        return f"\n{texts.TOOL_EXECUTING.format(tool=escape(tool_name), args=args_str)}"

    @staticmethod
    def tool_done_message(tool_name: str) -> str:
        return f"\n{texts.TOOL_DONE.format(tool=escape(tool_name))}\n"

    @staticmethod
    def tool_error_message(tool_name: str, error: str) -> str:
        return f"\n{texts.TOOL_ERROR.format(tool=escape(tool_name), error=escape(error))}\n"

    def process_stream_event(
        self,
        event: StreamEvent,
        history: list[Message],
        reply: str,
        display_content: str,
        temp_executing: str,
    ) -> tuple[str, str, str]:
        if event.type == EventType.CONTENT:
            content_chunk = event.content or ""
            reply = reply + content_chunk
            display_content = display_content + content_chunk
            return reply, display_content, temp_executing
        if event.type == EventType.TOOL_CALL_START:
            return (
                reply,
                display_content,
                self.tool_executing_message(event.tool_name, event.tool_args),
            )
        if event.type == EventType.TOOL_CALL_END:
            handle_checklist_tool_event(
                event.tool_name,
                history,
                renderer=self.render_helper.renderer if self.show_checklist else None,
                transcript=self.render_helper.transcript,
                agent_settings=self.agent_settings,
                session_start_mtime=self.session_started_at,
                render=self.show_checklist,
            )
            self.clear_progress()
            return reply, display_content + self.tool_done_message(event.tool_name), ""
        if event.type == EventType.TOOL_CALL_ERROR:
            self.clear_progress()
            return (
                reply,
                display_content + self.tool_error_message(event.tool_name, event.error or ""),
                "",
            )
        return reply, display_content, temp_executing


def build_stream_render_helper(
    *,
    transcript: list[str],
    renderer: ScreenRenderer,
    agent_settings: AgentSettings,
    session_started_at: float,
    show_checklist: bool,
) -> ReplRenderHelper:
    """Factory to create a configured `ReplRenderHelper`."""
    return ReplRenderHelper(
        transcript=transcript,
        renderer=renderer,
        agent_settings=agent_settings,
        session_started_at=session_started_at,
        show_checklist=show_checklist,
    )
