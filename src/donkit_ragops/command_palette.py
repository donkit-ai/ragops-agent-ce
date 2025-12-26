"""
Command palette module for RAGOps Agent CE.

Provides command palette functionality with autosuggest and filtering.
Activated when user presses '/' in the input box.
"""

import select
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

if TYPE_CHECKING:
    import termios
    import tty

# Import terminal manipulation utilities
try:
    import termios
    import tty

    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

try:
    import msvcrt

    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False

# Check if we can use interactive features
INTERACTIVE_AVAILABLE = TERMIOS_AVAILABLE or MSVCRT_AVAILABLE

console = Console()


@dataclass
class Command:
    """Represents a command in the command palette."""

    name: str  # e.g., "/help"
    description: str  # e.g., "Show help message"
    category: str  # e.g., "Navigation"
    template: str  # Text to insert, e.g., ":help"


class CommandRegistry:
    """Registry for managing commands in the command palette."""

    def __init__(self):
        self.commands: list[Command] = []
        self._register_default_commands()

    def _register_default_commands(self) -> None:
        """Register default commands."""
        # Navigation commands
        self.register(
            Command(
                name="/help",
                description="Show help message",
                category="Navigation",
                template=":help",
            )
        )
        self.register(
            Command(
                name="/clear",
                description="Clear conversation transcript",
                category="Navigation",
                template=":clear",
            )
        )
        self.register(
            Command(
                name="/exit",
                description="Exit the agent",
                category="Navigation",
                template=":quit",
            )
        )

        # Agent Control
        self.register(
            Command(
                name="/provider",
                description="Select LLM provider",
                category="Agent Control",
                template=":provider",
            )
        )
        self.register(
            Command(
                name="/model",
                description="Select LLM model",
                category="Agent Control",
                template=":model",
            )
        )

        # Project Management
        self.register(
            Command(
                name="/projects",
                description="List all projects",
                category="Project Management",
                template="List all projects",
            )
        )
        self.register(
            Command(
                name="/create-project",
                description="Create new RAG project",
                category="Project Management",
                template="Create a new RAG project",
            )
        )

        # System
        self.register(
            Command(
                name="/status",
                description="Show system status",
                category="System",
                template="Show system status",
            )
        )
        self.register(
            Command(
                name="/checklist",
                description="Show current checklist",
                category="System",
                template="Show current checklist",
            )
        )
        self.register(
            Command(
                name="/tools",
                description="List available tools",
                category="System",
                template="List available tools",
            )
        )

    def register(self, command: Command) -> None:
        """Register a command."""
        self.commands.append(command)

    def filter(self, query: str) -> list[Command]:
        """Filter commands by query string (matches name or description)."""
        if not query:
            return self.commands

        query_lower = query.lower()
        filtered = []
        for cmd in self.commands:
            if (
                query_lower in cmd.name.lower()
                or query_lower in cmd.description.lower()
                or query_lower in cmd.category.lower()
            ):
                filtered.append(cmd)
        return filtered


def _read_key_windows() -> str:
    """Read a key from Windows console."""
    if not MSVCRT_AVAILABLE:
        raise ImportError("msvcrt not available")

    if msvcrt.kbhit():
        ch = msvcrt.getch()
        # Handle special keys
        if ch in (b"\x00", b"\xe0"):  # Special key prefix
            ch2 = msvcrt.getch()
            # Arrow keys
            if ch2 == b"H":  # Up
                return "\x1b[A"
            elif ch2 == b"P":  # Down
                return "\x1b[B"
            elif ch2 == b"M":  # Right
                return "\x1b[C"
            elif ch2 == b"K":  # Left
                return "\x1b[D"
            return ""
        return ch.decode("utf-8", errors="ignore")
    return ""


def _read_key_unix() -> str:
    """Read a key from Unix terminal."""
    if not TERMIOS_AVAILABLE:
        raise ImportError("termios not available")

    if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
        return sys.stdin.read(1)
    return ""


class CommandPalette:
    """Handles command palette UI with filtering and navigation."""

    def __init__(self, registry: CommandRegistry, initial_query: str = ""):
        self.registry = registry
        self.query = initial_query
        self.selected_index = 0
        self.filtered_commands = self.registry.filter(self.query)

    def _highlight_match(self, text: str, query: str) -> Text:
        """Highlight matching text in command name."""
        if not query:
            return Text(text)

        text_lower = text.lower()
        query_lower = query.lower()

        if query_lower not in text_lower:
            return Text(text)

        result = Text()
        start = 0
        idx = text_lower.find(query_lower, start)
        while idx != -1:
            # Add text before match
            if idx > start:
                result.append(text[start:idx])
            # Add highlighted match
            result.append(text[idx : idx + len(query)], style="bold cyan")
            start = idx + len(query)
            idx = text_lower.find(query_lower, start)

        # Add remaining text
        if start < len(text):
            result.append(text[start:])

        return result

    def _create_palette_panel(self, selected_idx: int) -> Panel:
        """Create command palette panel with filtered commands."""
        content = Text()

        if not self.filtered_commands:
            content.append("No commands found", style="dim red")
            content.append("\n\n")
            content.append("Try a different search term", style="dim")
        else:
            # Limit to 10 visible commands for better UX
            visible_commands = self.filtered_commands[:10]
            for idx, cmd in enumerate(visible_commands):
                is_selected = idx == selected_idx

                # Selection indicator
                indicator = "❯ " if is_selected else "  "
                indicator_style = "bold cyan" if is_selected else "dim"
                content.append(indicator, style=indicator_style)

                # Command name with highlight
                name_text = self._highlight_match(cmd.name, self.query)
                if is_selected:
                    name_text.stylize(Style(bold=True))
                    name_text.stylize(Style(bgcolor="grey11"), 0, len(name_text))
                content.append_text(name_text)

                # Category badge
                content.append("  ", style="dim")
                content.append(f"[{cmd.category}]", style="dim cyan")

                # Description
                content.append("  ", style="dim")
                content.append(cmd.description, style="dim white")

                content.append("\n")

            if len(self.filtered_commands) > 10:
                content.append("\n", style="dim")
                content.append(
                    f"... and {len(self.filtered_commands) - 10} more",
                    style="dim",
                )

        # Add keyboard hints
        content.append("\n", style="")
        content.append("─" * 50, style="dim")
        content.append("\n")
        content.append("  ", style="")
        content.append("↑/↓", style="bold yellow")
        content.append(" Navigate  │  ", style="dim")
        content.append("Enter", style="bold green")
        content.append(" Insert  │  ", style="dim")
        content.append("Tab", style="bold cyan")
        content.append(" Complete  │  ", style="dim")
        content.append("Esc", style="bold red")
        content.append(" Cancel", style="dim")

        return Panel(
            content,
            title="[bold cyan]Command Palette[/bold cyan]",
            title_align="left",
            border_style="cyan",
            expand=False,
            padding=(1, 2),
        )

    def get_selection(self) -> Command | None:
        """
        Get user selection from command palette.

        Returns:
            Selected Command or None if cancelled
        """
        try:
            return self._interactive_palette()
        except (ImportError, OSError):
            return None

    def _interactive_palette(self) -> Command | None:
        """Interactive command palette with filtering."""
        if not sys.stdin.isatty() and not MSVCRT_AVAILABLE:
            raise ImportError("Not running in a terminal")

        # Update filtered commands
        self.filtered_commands = self.registry.filter(self.query)
        if self.selected_index >= len(self.filtered_commands):
            self.selected_index = 0

        # Setup terminal for Unix
        old_settings = None
        fd = None
        if TERMIOS_AVAILABLE:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        with Live(
            self._create_palette_panel(self.selected_index),
            console=console,
            refresh_per_second=20,
        ) as live:
            try:
                while True:
                    # Update filtered commands based on current query
                    self.filtered_commands = self.registry.filter(self.query)
                    if self.selected_index >= len(self.filtered_commands):
                        self.selected_index = max(0, len(self.filtered_commands) - 1)

                    live.update(self._create_palette_panel(self.selected_index))

                    # Read key (cross-platform)
                    if MSVCRT_AVAILABLE:
                        char = _read_key_windows()
                        if not char:
                            time.sleep(0.05)
                            continue
                    else:
                        char = _read_key_unix()
                        if not char:
                            continue

                    if char in ("\r", "\n"):  # Enter
                        if self.filtered_commands:
                            return self.filtered_commands[self.selected_index]
                        return None
                    elif char == "\t":  # Tab - auto-complete first match
                        if self.filtered_commands:
                            return self.filtered_commands[0]
                        return None
                    elif char == "\x03":  # Ctrl+C
                        return None
                    elif char == "\x04":  # Ctrl+D
                        return None
                    elif char == "\x1b":  # Escape sequence
                        # Check for Escape key (not arrow key)
                        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if not ready:
                            # Just Escape key
                            return None

                        # Might be arrow key, read next chars
                        next1 = sys.stdin.read(1) if ready else ""
                        if next1 == "[":
                            next2 = (
                                sys.stdin.read(1)
                                if select.select([sys.stdin], [], [], 0.01)[0]
                                else ""
                            )
                            if next2 == "A":  # Up arrow
                                if self.filtered_commands:
                                    self.selected_index = (self.selected_index - 1) % len(
                                        self.filtered_commands
                                    )
                            elif next2 == "B":  # Down arrow
                                if self.filtered_commands:
                                    self.selected_index = (self.selected_index + 1) % len(
                                        self.filtered_commands
                                    )
                    elif char in ("\x7f", "\b"):  # Backspace
                        if len(self.query) > 0:
                            self.query = self.query[:-1]
                            self.selected_index = 0
                    elif len(char) == 1 and ord(char) >= 32:  # Printable characters
                        self.query += char
                        self.selected_index = 0

            finally:
                if old_settings is not None and fd is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return None
