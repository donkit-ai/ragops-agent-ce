"""Helpers for rendering transcript and processing stream events in the REPL."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from donkit.llm import Message
from rich.markup import escape

from donkit_ragops import texts
from donkit_ragops.agent.agent import EventType, StreamEvent
from donkit_ragops.checklist_manager import get_active_checklist_text, handle_checklist_tool_event
from donkit_ragops.display import ScreenRenderer, print_checklist_inline
from donkit_ragops.schemas.agent_schemas import AgentSettings
from donkit_ragops.ui import get_ui


def format_timestamp() -> str:
    """Return timestamp string only in DEBUG mode.

    Returns empty string in production for cleaner output.
    """
    log_level = os.getenv("RAGOPS_LOG_LEVEL", "").upper()
    if log_level == "DEBUG":
        return "[dim]" + time.strftime("[%H:%M]", time.localtime()) + "[/] "
    return ""


def render_markdown_to_rich(text: str) -> str:
    """Convert Markdown text to Rich markup with extended support."""
    result = text

    # Step 1: Protect and render code blocks (```) first to avoid conflicts
    code_blocks = []

    def save_code_block(match):
        lang = match.group(1).strip() if match.group(1) else ""
        code = match.group(2)
        idx = len(code_blocks)
        # Use special character to avoid conflicts with markdown syntax
        placeholder = f"\x00CODEBLOCK{idx}\x00"

        # Escape code content to prevent Rich markup interpretation
        escaped_code = escape(code.rstrip("\n"))

        # Render code block with opening/closing markers and colored content
        if lang:
            rendered = f"[dim]```{lang}[/dim]\n[green]{escaped_code}[/green]\n[dim]```[/dim]"
        else:
            rendered = f"[dim]```[/dim]\n[green]{escaped_code}[/green]\n[dim]```[/dim]"

        code_blocks.append(rendered)
        return placeholder

    # Match ```lang\ncode\n``` or ```\ncode\n``` - support optional language
    result = re.sub(r"```([^\n]*)\n(.*?)```", save_code_block, result, flags=re.DOTALL)

    # Step 2: Inline code (must be before bold/italic to avoid conflicts)
    result = re.sub(r"`([^`\n]+)`", r"[cyan]\1[/cyan]", result)

    # Step 3: Strikethrough ~~text~~
    result = re.sub(r"~~([^~]+)~~", r"[dim strike]\1[/dim strike]", result)

    # Step 4: Bold and italic combinations
    # Bold+Italic: ***text*** or ___text___ (allow multiline with DOTALL)
    result = re.sub(r"\*\*\*(.+?)\*\*\*", r"[bold italic]\1[/bold italic]", result, flags=re.DOTALL)
    result = re.sub(r"___(.+?)___", r"[bold italic]\1[/bold italic]", result, flags=re.DOTALL)

    # Bold: **text** or __text__ (allow multiline with DOTALL)
    result = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", result, flags=re.DOTALL)
    result = re.sub(r"__(.+?)__", r"[bold]\1[/bold]", result, flags=re.DOTALL)

    # Italic: *text* or _text_ (but don't match list items or underscores in words)
    result = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"[italic]\1[/italic]", result)
    result = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"[italic]\1[/italic]", result)

    # Step 5: Headers with different levels and colors
    result = re.sub(r"^######\s+(.+)$", r"[bold]\1[/bold]", result, flags=re.MULTILINE)
    result = re.sub(r"^#####\s+(.+)$", r"[bold blue]\1[/bold blue]", result, flags=re.MULTILINE)
    result = re.sub(r"^####\s+(.+)$", r"[bold cyan]\1[/bold cyan]", result, flags=re.MULTILINE)
    result = re.sub(r"^###\s+(.+)$", r"[bold magenta]\1[/bold magenta]", result, flags=re.MULTILINE)
    result = re.sub(r"^##\s+(.+)$", r"[bold yellow]\1[/bold yellow]", result, flags=re.MULTILINE)
    result = re.sub(r"^#\s+(.+)$", r"[bold green]\1[/bold green]", result, flags=re.MULTILINE)

    # Step 6: Blockquotes > text
    result = re.sub(
        r"^>\s+(.+)$", r"[dim]▎[/dim] [italic dim]\1[/italic dim]", result, flags=re.MULTILINE
    )

    # Step 7: Horizontal rules ---, ***, ___
    result = re.sub(
        r"^(\s*)([-*_])\2{2,}\s*$", r"\1[dim]─────────────────[/dim]", result, flags=re.MULTILINE
    )

    # Step 8: Links [text](url) - show both text and url
    result = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"[link=\2]\1[/link] [dim](\2)[/dim]", result)

    # Step 9: List items with bullet points
    # Unordered: - text or * text -> • text (preserve indentation)
    result = re.sub(r"^(\s*)[-*]\s+", r"\1[yellow]•[/yellow] ", result, flags=re.MULTILINE)
    # Ordered: 1. text -> highlight number
    result = re.sub(r"^(\s*)(\d+)\.\s+", r"\1[cyan]\2.[/cyan] ", result, flags=re.MULTILINE)

    # Step 10: Restore code blocks
    for i, code_block in enumerate(code_blocks):
        result = result.replace(f"\x00CODEBLOCK{i}\x00", code_block)

    return result


@dataclass
class ReplRenderHelper:
    """Encapsulates transcript rendering and streaming utilities for the REPL.

    Redesigned for append-only output mode - no screen clearing.
    """

    transcript: list[str]
    renderer: ScreenRenderer
    agent_settings: AgentSettings
    session_started_at: float
    show_checklist: bool

    def render_current_screen(self, cl_text: str | None = None) -> None:
        """Show checklist inline (append-only mode).

        No longer clears screen - just prints checklist if needed.
        """
        if self.show_checklist:
            if cl_text is None:
                cl_text = get_active_checklist_text(self.session_started_at)
            print_checklist_inline(cl_text)

    def append_user_line(self, text: str) -> None:
        """Add user input line to transcript.

        Note: Don't print here - get_user_input() already shows the prompt with user input.
        """
        ts = format_timestamp()
        line = f"{ts}{texts.USER_PREFIX} {escape(text)}"
        self.transcript.append(line)

    def append_agent_message(self, content: str) -> None:
        """Print agent message directly (append-only)."""
        ts = format_timestamp()
        line = f"{ts}{texts.AGENT_PREFIX} {content}"
        self.transcript.append(line)

    def print_agent_prefix(self) -> None:
        """Print agent prefix for streaming output."""
        ts = format_timestamp()
        ui = get_ui()
        ui.print(f"\n{ts}{texts.AGENT_PREFIX} ", end="")

    def start_agent_placeholder(self) -> int:
        """Insert placeholder for streaming agent output."""
        ts = format_timestamp()
        self.transcript.append(f"{ts}{texts.AGENT_PREFIX} ")
        return len(self.transcript) - 1

    def set_agent_line(self, index: int, display_content: str, temp_executing: str) -> None:
        """Update transcript line with current streaming content."""
        ts = format_timestamp()
        self.transcript[index] = f"{ts}{texts.AGENT_PREFIX} {display_content}{temp_executing}"

    def append_error(self, message: str) -> None:
        """Print error message directly."""
        ts = format_timestamp()
        ui = get_ui()
        line = f"{ts}[bold red]Error:[/bold red] {message}"
        self.transcript.append(line)
        ui.print(line)


@dataclass
class MCPEventHandler:
    """Handles MCP progress updates and stream events.

    Redesigned for append-only output mode.
    """

    render_helper: ReplRenderHelper
    agent_settings: AgentSettings
    session_started_at: float
    show_checklist: bool
    progress_line_index: int | None = field(default=None, init=False)

    def progress_callback(self, progress: float, total: float | None, message: str | None) -> None:
        """Callback compatible with `MCPClient` progress updates.

        In append-only mode, just prints progress inline.
        """
        if total is not None:
            percentage = (progress / total) * 100
            progress_text = texts.PROGRESS_PERCENTAGE.format(
                percentage=percentage, message=message or ""
            )
        else:
            progress_text = texts.PROGRESS_GENERIC.format(progress=progress, message=message or "")

        # In append-only mode, just print progress (no screen clearing)
        if self.progress_line_index is None:
            self.render_helper.transcript.append(progress_text)
            self.progress_line_index = len(self.render_helper.transcript) - 1
            ui = get_ui()
            ui.print(progress_text)
        else:
            # Update transcript but don't reprint (avoid spam)
            self.render_helper.transcript[self.progress_line_index] = progress_text

    def clear_progress(self) -> None:
        """Clear progress tracking (but don't remove from console - append-only)."""
        if self.progress_line_index is not None:
            # In append-only mode, we can't remove printed lines
            # Just reset the index
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
            # Markdown will be rendered at the end when we have the full message
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
