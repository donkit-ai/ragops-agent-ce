"""
Interactive input module for RagOps Agent CE.

Provides interactive input box functionality with real-time typing inside Rich panels.
Follows Single Responsibility Principle - handles only user input interactions.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Unix-only imports
try:
    import termios
    import tty

    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

try:
    import readline  # noqa: F401

    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

console = Console()


class InteractiveInputBox:
    """Handles interactive input with real-time typing inside a Rich panel."""

    def __init__(self):
        self.current_text = ""
        self.cursor_pos = 0

    def _create_input_panel(self, text: str, cursor: int) -> Panel:
        """Create input panel with current text and cursor."""
        content = Text()
        content.append("you", style="bold blue")
        content.append("> ", style="bold blue")

        # Add text with cursor
        if cursor < len(text):
            content.append(text[:cursor], style="white")
            content.append("█", style="white")  # Cursor
            content.append(text[cursor:], style="white")
        elif not text:  # Add hint if empty
            content.append("T", style="black on white")
            content.append("ype your message... ", style="dim")
            content.append("(:q to quit)", style="yellow dim")
        else:
            content.append(text, style="white")
            content.append("█", style="white")  # Cursor at end

        return Panel(
            content,
            title="[dim]Input[/dim]",
            title_align="center",
            border_style="white",
            height=3,
            expand=True,
        )

    def get_input(self) -> str:
        """Get user input with interactive box or fallback to simple prompt."""
        try:
            return self._interactive_input()
        except (ImportError, OSError):
            # Fallback to simple input if terminal manipulation fails
            return self._fallback_input()

    def _interactive_input(self) -> str:
        """Interactive input box with real-time typing inside the box."""
        # Fallback if termios is not available (e.g., on Windows)
        if not TERMIOS_AVAILABLE:
            raise ImportError("termios module not available on this platform")

        self.current_text = ""
        self.cursor_pos = 0

        # Use Live for real-time updates
        with Live(self._create_input_panel("", 0), console=console, refresh_per_second=4) as live:
            try:
                # Get terminal settings
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)

                while True:
                    # Update display
                    live.update(self._create_input_panel(self.current_text, self.cursor_pos))

                    # Read character
                    char = sys.stdin.read(1)

                    if char == "\r" or char == "\n":  # Enter
                        break
                    elif char == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    elif char == "\x04":  # Ctrl+D (EOF)
                        raise KeyboardInterrupt
                    elif char == "\x7f" or char == "\b":  # Backspace
                        if self.cursor_pos > 0:
                            self.current_text = (
                                self.current_text[: self.cursor_pos - 1]
                                + self.current_text[self.cursor_pos :]
                            )
                            self.cursor_pos -= 1
                    elif char == "\x1b":  # Escape sequence (arrow keys)
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == "[":
                            if next2 == "D" and self.cursor_pos > 0:  # Left arrow
                                self.cursor_pos -= 1
                            elif next2 == "C" and self.cursor_pos < len(
                                self.current_text
                            ):  # Right arrow
                                self.cursor_pos += 1
                    elif len(char) == 1 and ord(char) >= 32:  # Printable character
                        self.current_text = (
                            self.current_text[: self.cursor_pos]
                            + char
                            + self.current_text[self.cursor_pos :]
                        )
                        self.cursor_pos += 1

            except KeyboardInterrupt:
                raise
            finally:
                # Restore terminal settings
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return self.current_text.strip()

    def _fallback_input(self) -> str:
        """Fallback to simple input for incompatible terminals."""
        console.print()
        console.print("[bold blue]you>[/bold blue] ", end="")
        try:
            user_input = input().strip()
            return user_input
        except (EOFError, KeyboardInterrupt):
            raise


def get_user_input() -> str:
    """
    Main function to get user input.

    Returns:
        str: User input text (stripped of whitespace)

    Raises:
        KeyboardInterrupt: When user presses Ctrl+C or Ctrl+D
    """
    if TERMIOS_AVAILABLE:
        input_box = InteractiveInputBox()
        return input_box.get_input()

    # Fallback for incompatible terminals
    console.print("[bold blue]you>[/bold blue] ", end="")
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        raise
