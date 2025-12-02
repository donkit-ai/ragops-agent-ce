from __future__ import annotations

import os
import shutil
import sys

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ragops_agent_ce.schemas.agent_schemas import AgentSettings

"""
Display and rendering module for RAGOps Agent CE.

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
    )


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


def create_transcript_panel(
    transcript_lines: list[str],
    width: int,
    height: int | None = None,
    title: str = "Conversation",
) -> Panel:
    """
    Create a Rich panel for conversation transcript.

    Args:
        transcript_lines: List of transcript messages
        title: Panel title

    Returns:
        Panel: Rich panel containing the transcript
    """
    lines = transcript_lines or ["[dim italic]No messages yet. Start by typing a message![/]"]

    # wrapped_lines: list[Text] = []
    # for line in lines:
    #     rich_text = Text.from_markup(line)
    #     wrapped_parts = rich_text.wrap(console, width - 2)
    #     wrapped_lines.extend(wrapped_parts or [Text("")])

    # visible_parts = wrapped_lines[-(height - 4) :]

    # Combine lines into panel content
    content = Text()
    for line in lines:
        content.append("\n")
        content.append(Text.from_markup(line))

    # content = Align(clipped_text, align="left", vertical="bottom")

    kwargs = {}
    if height is not None:
        kwargs["height"] = height

    return Panel(
        content,
        title=f"[bold blue]{title}[/bold blue]",
        title_align="center",
        border_style="dim",
        expand=True,
        padding=(0, 1),
        **kwargs,
    )


def create_status_panel(agent_settings: AgentSettings, title: str = "Agent settings") -> Panel:
    status_lines = [f"LLM provider: [cyan]{agent_settings.llm_provider.name}[/cyan]"]
    if agent_settings.model:
        status_lines.append(f"Model: [cyan]{agent_settings.model}[/cyan]")
    return Panel(
        Text.from_markup("\n".join(status_lines)),
        title=f"[bold blue]{title}[/bold blue]",
        border_style="cyan",
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

    @staticmethod
    def render_project(
        transcript: list[str],
        checklist_text: str | None,
        agent_settings: AgentSettings,
    ) -> None:
        """Render conversation with a checklist panel on the right."""
        clear_screen_aggressive()

        width, height = shutil.get_terminal_size()
        height -= 3  # Reserve space for input prompt
        right_width = 50

        right_panels = [create_status_panel(agent_settings)]
        if checklist_text:
            checklist_panel = create_checklist_panel(checklist_text)
            right_panels.append(checklist_panel)

        table = Table(show_header=False, show_edge=False, padding=(0, 0), box=None, expand=True)
        table.add_column(width=width - right_width)
        table.add_column(width=right_width)
        table.add_row(
            create_transcript_panel(
                transcript,
                width=width - right_width,
            ),
            Align(Group(*right_panels, fit=False), vertical="bottom"),
        )

        console.print(table)

    @staticmethod
    def render_startup_screen() -> None:
        """Render the initial startup screen."""
        clear_screen_aggressive()

        console.print()
        console.print("[bold blue]🤖 RAGOps Agent CE[/bold blue]")
        console.print("[dim]Interactive AI Agent for RAG Operations[/dim]")
        console.print()
        console.print("[yellow]Commands:[/yellow]")
        console.print("  [bold]:help[/bold] - Show help")
        console.print("  [bold]:q[/bold] or [bold]:quit[/bold] - Exit")
        console.print(
            "  [bold]:agent <llm_provider>/<model>[/bold] - Change agent LLM provider and model"
        )
        console.print("  [bold]:clear[/bold] - Clear conversation")
        console.print()
        console.print("[dim]Type your message and press Enter to start...[/dim]")
        console.print()

    @staticmethod
    def render_goodbye_screen() -> None:
        """Render the goodbye screen on exit."""
        console.print()
        console.print("[bold blue]👋 Goodbye![/bold blue]")
        console.print("[dim]Thanks for using RAGOps Agent CE[/dim]")
        console.print()
