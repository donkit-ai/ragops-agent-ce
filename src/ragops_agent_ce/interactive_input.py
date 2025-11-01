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

# Windows-only imports
try:
    import msvcrt

    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False

try:
    import readline  # noqa: F401

    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

# Check if we can use interactive features
INTERACTIVE_AVAILABLE = TERMIOS_AVAILABLE or MSVCRT_AVAILABLE

console = Console()


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
            content.append("(:q to quit, :help to list commands)", style="yellow dim")
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


class InteractiveSelect:
    """Handles interactive selection menu with arrow key navigation."""

    def __init__(
        self,
        choices: list[str],
        title: str = "Select an option",
        default_index: int | None = None,
    ):
        self.choices = choices
        self.title = title
        self.selected_index = (
            default_index
            if default_index is not None and 0 <= default_index < len(choices)
            else 0
        )

    def _create_select_panel(self, selected_idx: int) -> Panel:
        """Create selection panel with choices and highlighted selection."""
        from rich.markup import escape

        content = Text()

        for idx, choice in enumerate(self.choices):
            if idx == selected_idx:
                # Highlighted selection with better visual indicator
                content.append("▶ ", style="bold cyan")
                # Parse Rich markup from choice string if present
                try:
                    if "[" in choice and "]" in choice:
                        # Create a temporary Text object to parse markup
                        temp_text = Text.from_markup(choice)
                        # Apply highlight styling while preserving original styles
                        for segment in temp_text:
                            # Combine original style with highlight
                            # Preserve colors but add cyan background for visibility
                            if segment.style:
                                # Try to extract color from original style
                                base_style = str(segment.style)
                                # Apply highlight with subtle background
                                if "green" in base_style or "bold green" in base_style:
                                    content.append(segment.text, style="bold green on #004400")
                                elif "yellow" in base_style:
                                    content.append(segment.text, style="bold yellow on #444400")
                                elif "cyan" in base_style:
                                    content.append(segment.text, style="bold cyan on #004444")
                                elif "dim" in base_style or "italic" in base_style:
                                    content.append(segment.text, style="dim on #222222")
                                elif "on black" in base_style:
                                    # Badge style - enhance with brighter background
                                    content.append(segment.text, segment.style)
                                else:
                                    content.append(segment.text, style="bold white on #222222")
                            else:
                                content.append(segment.text, style="bold white on #222222")
                    else:
                        content.append(choice, style="bold white on #222222")
                except Exception:
                    # Fallback if markup parsing fails
                    content.append(escape(choice), style="bold white on #222222")
            else:
                content.append("  ", style="dim")
                # Parse Rich markup for non-selected items - preserve original styles
                try:
                    if "[" in choice and "]" in choice:
                        temp_text = Text.from_markup(choice)
                        for segment in temp_text:
                            # Preserve original styling from markup
                            content.append(segment.text, style=segment.style or "white")
                    else:
                        content.append(choice, style="white")
                except Exception:
                    content.append(escape(choice), style="white")
            content.append("\n")

        # Add hint with better styling
        content.append("\n")
        content.append("─" * 60, style="dim")
        content.append("\n")
        content.append("  ", style="")
        content.append("↑/↓", style="bold yellow")
        content.append(" Navigate  │  ", style="dim")
        content.append("Enter", style="bold green")
        content.append(" Select  │  ", style="dim")
        content.append("Ctrl+C", style="bold red")
        content.append(" Cancel", style="dim")

        return Panel(
            content,
            title=f"[bold cyan]{self.title}[/bold cyan]",
            title_align="left",
            border_style="cyan",
            expand=True,
            padding=(1, 2),
        )

    def get_selection(self) -> str | None:
        """
        Get user selection with arrow keys or fallback to numbered input.

        Returns:
            Selected choice string or None if cancelled
        """
        try:
            return self._interactive_select()
        except (ImportError, OSError):
            # Fallback to numbered selection
            return self._fallback_select()

    def _interactive_select(self) -> str | None:
        if not sys.stdin.isatty() and not MSVCRT_AVAILABLE:
            raise ImportError("Not running in a terminal")

        # Use the initial selected_index from __init__
        initial_index = self.selected_index

        # Setup terminal for Unix
        old_settings = None
        if TERMIOS_AVAILABLE:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        with Live(
            self._create_select_panel(initial_index),
            console=console,
            refresh_per_second=20,
        ) as live:
            try:
                while True:
                    live.update(self._create_select_panel(self.selected_index))

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
                        return self.choices[self.selected_index]
                    elif char == "\x03":  # Ctrl+C
                        return None
                    elif char == "\x04":  # Ctrl+D
                        return None
                    # Handle arrow keys (converted to ANSI escape sequences)
                    elif char == "\x1b[A":  # Up
                        self.selected_index = (self.selected_index - 1) % len(self.choices)
                    elif char == "\x1b[B":  # Down
                        self.selected_index = (self.selected_index + 1) % len(self.choices)
                    elif char == "\x1b" and not MSVCRT_AVAILABLE:  # Unix arrow keys
                        # Unix: read next chars
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == "[":
                            if next2 == "A":  # Up arrow
                                self.selected_index = (self.selected_index - 1) % len(
                                    self.choices
                                )
                            elif next2 == "B":  # Down arrow
                                self.selected_index = (self.selected_index + 1) % len(
                                    self.choices
                                )

            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _fallback_select(self) -> str | None:
        """Fallback to numbered selection for incompatible terminals."""
        from rich.markup import escape

        console.print()
        console.print(f"[bold]{self.title}[/bold]")
        for idx, choice in enumerate(self.choices, 1):
            # Try to render Rich markup, fallback to plain text
            try:
                from rich.text import Text
                choice_text = Text.from_markup(choice)
                console.print(f"  {idx}. ", end="")
                console.print(choice_text)
            except Exception:
                console.print(f"  {idx}. {escape(choice)}")
        console.print()

        while True:
            try:
                console.print("[bold cyan]Enter number (or 'q' to cancel):[/bold cyan] ", end="")
                user_input = input().strip()

                if user_input.lower() in ("q", "quit", "cancel"):
                    return None

                choice_num = int(user_input)
                if 1 <= choice_num <= len(self.choices):
                    return self.choices[choice_num - 1]
                else:
                    console.print(
                        f"[red]Please enter a number between 1 and {len(self.choices)}[/red]"
                    )
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")
            except (EOFError, KeyboardInterrupt):
                return None


class InteractiveConfirm:
    """Handles interactive yes/no confirmation with arrow key navigation."""

    def __init__(self, question: str, default: bool = True):
        self.question = question
        self.default = default
        self.selected_yes = default

    def _create_confirm_panel(self, selected_yes: bool) -> Panel:
        """Create confirmation panel with yes/no options."""
        content = Text()
        content.append(self.question, style="white")
        content.append("\n\n")

        # Yes option
        if selected_yes:
            content.append("❯ ", style="bold green")
            content.append("Yes", style="bold green on black")
        else:
            content.append("  ", style="dim")
            content.append("Yes", style="white")

        content.append("  ")

        # No option
        if not selected_yes:
            content.append("❯ ", style="bold red")
            content.append("No", style="bold red on black")
        else:
            content.append("  ", style="dim")
            content.append("No", style="white")

        content.append("\n\n", style="dim")
        content.append("←/→: Navigate  ", style="yellow dim")
        content.append("Enter: Select  ", style="green dim")
        content.append("y/n: Quick select", style="cyan dim")

        return Panel(
            content,
            title="[bold]Confirm[/bold]",
            title_align="left",
            border_style="yellow",
            expand=False,
        )

    def get_confirmation(self) -> bool | None:
        """
        Get user confirmation with arrow keys or fallback to y/n input.

        Returns:
            True for yes, False for no, None if cancelled
        """
        try:
            return self._interactive_confirm()
        except (ImportError, OSError):
            return self._fallback_confirm()

    def _interactive_confirm(self) -> bool | None:
        if not sys.stdin.isatty() and not MSVCRT_AVAILABLE:
            raise ImportError("Not running in a terminal")

        self.selected_yes = self.default

        # Setup terminal for Unix
        old_settings = None
        if TERMIOS_AVAILABLE:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        with Live(
            self._create_confirm_panel(self.default),
            console=console,
            refresh_per_second=20,
        ) as live:
            try:
                while True:
                    live.update(self._create_confirm_panel(self.selected_yes))

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
                        return self.selected_yes
                    elif char in ("y", "Y"):  # Quick yes
                        return True
                    elif char in ("n", "N"):  # Quick no
                        return False
                    elif char == "\x03":  # Ctrl+C
                        return None
                    elif char == "\x04":  # Ctrl+D
                        return None
                    # Handle arrow keys (converted to ANSI escape sequences)
                    elif char in ("\x1b[C", "\x1b[D"):  # Right or Left
                        self.selected_yes = not self.selected_yes
                    elif char == "\x1b" and not MSVCRT_AVAILABLE:  # Unix arrow keys
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == "[":
                            if next2 in ("C", "D"):  # Right or Left arrow
                                self.selected_yes = not self.selected_yes

            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _fallback_confirm(self) -> bool | None:
        """Fallback to y/n input for incompatible terminals."""
        console.print()
        default_str = "Y/n" if self.default else "y/N"
        console.print(f"[bold]{self.question}[/bold] [{default_str}]: ", end="")

        try:
            user_input = input().strip().lower()

            if not user_input:
                return self.default
            elif user_input in ("y", "yes"):
                return True
            elif user_input in ("n", "no"):
                return False
            else:
                # Invalid input, use default
                return self.default
        except (EOFError, KeyboardInterrupt):
            return None


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


def interactive_select(
    choices: list[str],
    title: str = "Select an option",
    default_index: int | None = None,
) -> str | None:
    """
    Show interactive selection menu with arrow key navigation.

    Args:
        choices: List of options to choose from
        title: Title for the selection menu
        default_index: Optional initial selection index

    Returns:
        Selected choice string or None if cancelled
    """
    if INTERACTIVE_AVAILABLE:
        selector = InteractiveSelect(choices, title, default_index=default_index)
        return selector.get_selection()

    # Fallback for incompatible terminals
    selector = InteractiveSelect(choices, title, default_index=default_index)
    return selector._fallback_select()


def interactive_confirm(question: str, default: bool = True) -> bool | None:
    """
    Show interactive yes/no confirmation with arrow key navigation.

    Args:
        question: Question to ask the user
        default: Default value (True for Yes, False for No)

    Returns:
        True for yes, False for no, None if cancelled
    """
    if INTERACTIVE_AVAILABLE:
        confirmer = InteractiveConfirm(question, default)
        return confirmer.get_confirmation()

    # Fallback for incompatible terminals
    confirmer = InteractiveConfirm(question, default)
    return confirmer._fallback_confirm()
