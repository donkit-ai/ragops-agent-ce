from __future__ import annotations

import os
import shutil
import sys

from rich.align import Align
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

"""
Display and rendering module for RagOps Agent CE.

Handles screen rendering, panel creation, and terminal display logic.
Follows Single Responsibility Principle - manages only display-related operations.
"""


console = Console()


def create_checklist_panel(checklist_text: str | None, *, title: str = "Checklist") -> Panel:
    """
    Create a Rich panel for the checklist content.

    Args:
        checklist_text: Rich-markup string of the checklist
        title: Panel title

    Returns:
        Panel: Rich panel containing the checklist
    """
    if not checklist_text:
        content = Text("No checklist available", style="dim")
    else:
        content = Text.from_markup(checklist_text)
    # Ensure wrapping inside the panel
    content.overflow = "fold"
    return Panel(
        content,
        title=f"[bold blue]{title}[/bold blue]",
        title_align="center",
        border_style="cyan",
        expand=True,
        padding=(0, 1),
    )


def render_screen_with_checklist(
    transcript_lines: list[str],
    checklist_text: str | None = None,
    *,
    show_prompt: bool = False,
) -> None:
    """
    Clear screen and render conversation panel alongside checklist panel.

    Args:
        transcript_lines: Conversation transcript
        checklist_text: Checklist content (Rich markup)
        show_prompt: Reserve space for input prompt
    """
    clear_screen_aggressive()

    if not checklist_text:
        render_screen(transcript_lines, show_prompt=show_prompt)
        return

    conv_panel = create_transcript_panel(transcript_lines, title="Conversation")
    checklist_panel = create_checklist_panel(checklist_text, title="Checklist")

    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(ratio=7)
    table.add_column(ratio=3)
    table.add_row(conv_panel, Align(checklist_panel, align="center", vertical="bottom"))
    console.print(Padding(table, (1, 0, 0, 0)))


def clear_screen_aggressive() -> None:
    """
    Perform aggressive screen clearing to completely remove old content.

    Skips clearing when RAGOPS_LOG_LEVEL=DEBUG to preserve debug logs.

    Uses multiple methods for maximum terminal compatibility:
    - Rich console clear
    - ANSI escape sequences for screen clearing
    - Full terminal reset
    """
    # Skip aggressive clearing if DEBUG logging is enabled to preserve logs
    log_level = os.getenv("RAGOPS_LOG_LEVEL", "").upper()
    if log_level == "DEBUG":
        return

    # Rich console clear
    console.clear()

    # Additional terminal clearing for better compatibility
    if hasattr(console, "_file") and hasattr(console._file, "write"):
        # Clear screen and move cursor to home position
        console._file.write("\033[2J\033[H")
        console._file.flush()

    # Alternative approach using print for maximum compatibility
    print("\033c", end="")  # Full terminal reset
    sys.stdout.flush()


def create_transcript_panel(transcript_lines: list[str], title: str = "RagOpsAgent CE") -> Panel:
    """
    Create a Rich panel for conversation transcript.

    Args:
        transcript_lines: List of transcript messages
        title: Panel title

    Returns:
        Panel: Rich panel containing the transcript
    """
    lines = transcript_lines or ["[dim italic]No messages yet. Start by typing a message![/]"]
    width, _ = shutil.get_terminal_size()

    # Effective text width inside panel (minus padding and borders)
    inner_width = max(width - 4, 10)  # 2 borders + 2 padding (approx.)

    wrapped_lines: list[Text] = []
    for line in lines:
        rich_text = Text.from_markup(line)
        wrapped_parts = rich_text.wrap(console, inner_width)
        wrapped_lines.extend(wrapped_parts or [Text("")])

    # Show ALL lines - no height restriction, terminal will handle scrolling
    # Combine lines into panel content
    content = Text()
    for i, part in enumerate(wrapped_lines):
        if i > 0:
            content.append("\n")
        content.append(part)

    return Panel(
        content,
        title=f"[bold blue]{title}[/bold blue]",
        title_align="center",
        border_style="green",
        expand=True,
        padding=(0, 1),
    )


def render_screen(transcript_lines: list[str], *, show_prompt: bool = False) -> None:
    """
    Clear screen completely and render conversation panel.

    Args:
        transcript_lines: List of transcript messages
        show_prompt: Whether to reserve space for input prompt (affects panel height)
    """
    clear_screen_aggressive()
    console.print(
        Padding(
            create_transcript_panel(transcript_lines, title="Conversation"),
            (1, 0, 0, 0),
        )
    )


def print_message(message: str, style: str = "") -> None:
    """
    Print a message with optional Rich styling.

    Args:
        message: Message text
        style: Rich style string (e.g., "bold red", "dim")
    """
    if style:
        console.print(f"[{style}]{message}[/{style}]")
    else:
        console.print(message)


def print_error(message: str) -> None:
    """
    Print an error message with red styling.

    Args:
        message: Error message
    """
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """
    Print a success message with green styling.

    Args:
        message: Success message
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """
    Print a warning message with yellow styling.

    Args:
        message: Warning message
    """
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """
    Print an info message with blue styling.

    Args:
        message: Info message
    """
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


class ScreenRenderer:
    """
    High-level screen rendering manager.

    Manages the overall screen layout and rendering coordination.
    """

    def __init__(self):
        self.last_transcript_size = 0

    def render_conversation_screen(
        self, transcript: list[str], show_input_space: bool = False
    ) -> None:
        """
        Render the main conversation screen.

        Args:
            transcript: Conversation transcript lines
            show_input_space: Whether to reserve space for input box
        """
        render_screen(transcript, show_prompt=show_input_space)
        self.last_transcript_size = len(transcript)

    def render_conversation_and_checklist(
        self,
        transcript: list[str],
        checklist_text: str | None,
        show_input_space: bool = False,
    ) -> None:
        """Render conversation with a checklist panel on the right."""
        render_screen_with_checklist(transcript, checklist_text, show_prompt=show_input_space)
        self.last_transcript_size = len(transcript)

    def should_rerender(self, transcript: list[str]) -> bool:
        """
        Check if screen needs re-rendering based on transcript changes.

        Args:
            transcript: Current transcript

        Returns:
            bool: True if re-rendering is needed
        """
        return len(transcript) != self.last_transcript_size

    def render_startup_screen(self) -> None:
        """Render the initial startup screen."""
        clear_screen_aggressive()

        console.print()
        console.print("[bold blue]🤖 RagOps Agent CE[/bold blue]")
        console.print("[dim]Interactive AI Agent for RAG Operations[/dim]")
        console.print()
        console.print("[yellow]Commands:[/yellow]")
        console.print("  [bold]:help[/bold] - Show help")
        console.print("  [bold]:q[/bold] or [bold]:quit[/bold] - Exit")
        console.print("  [bold]:clear[/bold] - Clear conversation")
        console.print()
        console.print("[dim]Type your message and press Enter to start...[/dim]")
        console.print()

    def render_goodbye_screen(self) -> None:
        """Render the goodbye screen on exit."""
        console.print()
        console.print("[bold blue]👋 Goodbye![/bold blue]")
        console.print("[dim]Thanks for using RagOps Agent CE[/dim]")
        console.print()
