"""
Interactive input module for RagOps Agent CE.

Provides interactive input box functionality with real-time typing inside Rich panels.
Follows Single Responsibility Principle - handles only user input interactions.
"""
import select
import sys
import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    import termios
    import tty


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
        self.cursor_visible = True
        self.last_blink = time.time()
        self.blink_interval = 0.5  # seconds

    def _create_input_panel(self, text: str, cursor: int, show_cursor: bool) -> Panel:
        """Create input panel with current text and cursor."""
        content = Text()
        content.append("you", style="bold blue")
        content.append("> ", style="bold blue")

        if cursor < len(text):
            content.append(text[:cursor], style="white")
            if show_cursor:
                # Blinking cursor
                content.append(text[cursor], style="black on white")
            else:
                content.append(text[cursor], style="white")
            content.append(text[cursor + 1 :], style="white")
        elif not text:
            if show_cursor:
                content.append("T", style="black on white")
                content.append("ype your message... ", style="dim")
            else:
                content.append("Type your message... ", style="dim")
            content.append("(:q to quit)", style="yellow dim")
        else:
            content.append(text, style="white")
            if show_cursor:
                content.append("█", style="white")

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
        if not sys.stdin.isatty():
            raise ImportError("Not running in a terminal")

        self.current_text = ""
        self.cursor_pos = 0
        self.cursor_visible = True
        self.last_blink = time.time()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        with Live(
            self._create_input_panel("", 0, True),
            console=console,
            refresh_per_second=20,
        ) as live:
            try:
                while True:
                    # Blink cursor
                    now = time.time()
                    if now - self.last_blink >= self.blink_interval:
                        self.cursor_visible = not self.cursor_visible
                        self.last_blink = now

                    live.update(
                        self._create_input_panel(
                            self.current_text, self.cursor_pos, self.cursor_visible
                        )
                    )

                    # Check input
                    if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                        char = sys.stdin.read(1)
                    else:
                        continue

                    if char in ("\r", "\n"):  # Enter
                        break
                    elif char == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    elif char == "\x04":  # Ctrl+D
                        raise KeyboardInterrupt
                    elif char in ("\x7f", "\b"):  # Backspace
                        if self.cursor_pos > 0:
                            self.current_text = (
                                self.current_text[: self.cursor_pos - 1]
                                + self.current_text[self.cursor_pos :]
                            )
                            self.cursor_pos -= 1
                    elif char == "\x1b":  # Arrows
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == "[":
                            if next2 == "D" and self.cursor_pos > 0:
                                self.cursor_pos -= 1
                            elif next2 == "C" and self.cursor_pos < len(self.current_text):
                                self.cursor_pos += 1
                    elif len(char) == 1 and ord(char) >= 32:
                        self.current_text = (
                            self.current_text[: self.cursor_pos]
                            + char
                            + self.current_text[self.cursor_pos :]
                        )
                        self.cursor_pos += 1

            finally:
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
