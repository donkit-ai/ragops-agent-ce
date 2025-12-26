"""
Display and rendering module for RAGOps Agent CE.

Provides simple append-only output functions for the CLI.
Follows Single Responsibility Principle - manages only display-related operations.
"""

from __future__ import annotations

from donkit_ragops import __version__
from donkit_ragops.prints import format_header
from donkit_ragops.schemas.agent_schemas import AgentSettings
from donkit_ragops.ui import get_ui
from donkit_ragops.ui.styles import StyleName, styled_text


def print_message(message: str, style: StyleName | None = None) -> None:
    """
    Print a message with optional styling.

    Args:
        message: Message text
        style: Optional StyleName for semantic styling
    """
    ui = get_ui()
    ui.print(message, style)


def print_error(message: str) -> None:
    """
    Print an error message with red styling.

    Args:
        message: Error message
    """
    ui = get_ui()
    ui.print_error(message)


def print_success(message: str) -> None:
    """
    Print a success message with green styling.

    Args:
        message: Success message
    """
    ui = get_ui()
    ui.print_success(message)


def print_warning(message: str) -> None:
    """
    Print a warning message with yellow styling.

    Args:
        message: Warning message
    """
    ui = get_ui()
    ui.print_warning(message)


def print_info(message: str) -> None:
    """
    Print an info message with blue styling.

    Args:
        message: Info message
    """
    ui = get_ui()
    ui.print_info(message)


def print_checklist_inline(checklist_text: str | None) -> None:
    """
    Print checklist inline without panel wrapper.

    Args:
        checklist_text: Rich-markup string of the checklist
    """
    if not checklist_text:
        return
    ui = get_ui()
    ui.newline()
    ui.print_styled(styled_text((StyleName.DIM, "── Checklist ──")))
    # For checklist content, use markdown rendering (handles Rich markup)
    ui.print_markdown(checklist_text)


def print_startup_header(provider: str, model: str | None = None) -> None:
    """
    Print compact startup header.

    Args:
        provider: LLM provider name
        model: Optional model name
    """
    ui = get_ui()
    ui.newline()
    ui.print_styled(format_header(__version__, provider, model))
    ui.newline()


class ScreenRenderer:
    """
    High-level screen rendering manager.

    Simplified for append-only output mode.
    """

    @staticmethod
    def render_startup_screen(agent_settings: AgentSettings | None = None) -> None:
        """Render the initial startup screen with compact header."""
        ui = get_ui()
        if agent_settings:
            print_startup_header(
                provider=agent_settings.llm_provider.name,
                model=agent_settings.model,
            )
        else:
            ui.newline()
            ui.print_styled(format_header(__version__, "initializing", None))
            ui.newline()

        ui.print("Type your message and press Enter to start...", StyleName.DIM)
        ui.print("Commands: :help, :q, :clear, :provider, :model", StyleName.DIM)
        ui.newline()

    @staticmethod
    def render_goodbye_screen() -> None:
        """Render the goodbye screen on exit."""
        ui = get_ui()
        ui.newline()
        ui.print_styled(styled_text((StyleName.INFO, "👋 Goodbye!")))
        ui.print("Thanks for using RAGOps Agent CE", StyleName.DIM)
        ui.newline()


# Legacy compatibility - deprecated functions
# These are kept for backward compatibility but should not be used in new code


def create_checklist_panel(checklist_text: str | None, *, title: str = "Checklist"):
    """
    DEPRECATED: Use print_checklist_inline() instead.

    Creates a simple text representation for backward compatibility.
    """
    from rich.panel import Panel
    from rich.text import Text

    if not checklist_text:
        content = Text("No checklist available", style="dim")
    else:
        content = Text.from_markup(checklist_text)
    content.overflow = "fold"
    return Panel(
        content,
        title=f"[bold blue]{title}[/bold blue]",
        title_align="center",
        border_style="cyan",
        expand=True,
    )
