"""
Interactive input module for RAGOps Agent CE.

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
from rich.style import Style
from rich.text import Text

from .command_palette import CommandPalette
from .command_palette import CommandRegistry

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
        self.command_registry = CommandRegistry()
        self.palette_active = False
        self.palette_query = ""
        self.palette_start_pos = 0
        self.palette_selected_index = 0

    def _get_cursor_position(self, text: str, cursor: int) -> tuple[int, int]:
        """Get cursor position as (line, column) from absolute cursor position."""
        lines = text[:cursor].split("\n")
        line = len(lines) - 1
        column = len(lines[-1]) if lines else 0
        return line, column

    def _get_text_lines(self, text: str) -> list[str]:
        """Split text into lines, handling empty text."""
        if not text:
            return [""]
        return text.split("\n")

    def _calculate_panel_height(self, text: str) -> int:
        """Calculate panel height based on number of lines."""
        lines = self._get_text_lines(text)
        num_lines = len(lines)
        # Base height: 3 (border + padding), plus 1 for each line of text
        # Add extra height for prompt and hint text if empty
        if not text:
            height = 5  # Space for prompt, hint, and cursor
        else:
            # 1 for top padding, 1 for bottom padding, plus lines of text
            height = max(4, num_lines + 3)
        # Cap at reasonable maximum
        return min(height, 25)

    def _create_input_panel(self, text: str, cursor: int, show_cursor: bool) -> Panel:
        """Create input panel with current text and cursor, supporting multiline."""
        content = Text()
        content.append("you", style="bold blue")
        content.append("> ", style="bold blue")

        lines = self._get_text_lines(text)
        line, col = self._get_cursor_position(text, cursor)

        if not text:
            # Empty text placeholder
            if show_cursor:
                content.append("T", style="black on white")
                content.append("ype your message... ", style="dim")
            else:
                content.append("Type your message... ", style="dim")
            content.append(
                "(Enter to submit, Alt+Enter for newline, :q to quit)", style="yellow dim"
            )
        else:
            # Render multiline text with cursor
            for line_idx, line_text in enumerate(lines):
                if line_idx > 0:
                    # New line - add continuation indicator
                    content.append("\n")
                    content.append("└─ ", style="dim cyan")  # Visual continuation indicator

                if line_idx == line:
                    # Current line with cursor
                    if col < len(line_text):
                        # Cursor in middle of line
                        content.append(line_text[:col], style="white")
                        if show_cursor:
                            content.append(line_text[col], style="black on white")
                        else:
                            content.append(line_text[col], style="white")
                        content.append(line_text[col + 1 :], style="white")
                    else:
                        # Cursor at end of line
                        content.append(line_text, style="white")
                        if show_cursor:
                            content.append("█", style="white")
                        else:
                            content.append(" ", style="white")  # Space for cursor
                else:
                    # Other lines without cursor
                    content.append(line_text, style="dim white")

        # Calculate dynamic height
        height = self._calculate_panel_height(text)

        return Panel(
            content,
            title="[dim]Input[/dim]",
            title_align="center",
            border_style="white",
            height=height,
            expand=True,
        )

    def _should_activate_palette(self, text: str, cursor_pos: int) -> bool:
        """
        Check if command palette should be activated
        (when / is at start of line or after space).
        """
        if cursor_pos == 0:
            return True  # At start of line
        if cursor_pos > 0 and text[cursor_pos - 1] in (" ", "\n"):
            return True  # After space or newline
        return False

    def _create_combined_panel(self) -> Panel:
        """Create combined panel with palette overlay and input box."""
        from rich.layout import Layout

        if self.palette_active:
            filtered = self.command_registry.filter(self.palette_query)
            # Show palette even if no matches (will show "No commands found" message)
            palette = CommandPalette(self.command_registry, self.palette_query)
            palette.selected_index = self.palette_selected_index
            # Ensure selected_index is within bounds
            if filtered:
                if self.palette_selected_index >= len(filtered):
                    self.palette_selected_index = len(filtered) - 1
            else:
                self.palette_selected_index = 0
            palette.selected_index = self.palette_selected_index
            palette_panel = palette._create_palette_panel(self.palette_selected_index)
            # Create a layout with palette on top and input below
            layout = Layout()
            # Calculate palette height: at least 6 lines, up to 15 lines
            palette_height = min(15, max(6, len(filtered) + 4) if filtered else 6)
            layout.split_column(
                Layout(palette_panel, size=palette_height),
                Layout(
                    self._create_input_panel(
                        self.current_text, self.cursor_pos, self.cursor_visible
                    )
                ),
            )
            return layout
        return self._create_input_panel(self.current_text, self.cursor_pos, self.cursor_visible)

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
        self.palette_active = False
        self.palette_query = ""
        self.palette_start_pos = 0
        self.palette_selected_index = 0

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

                    live.update(self._create_combined_panel())

                    # Check input - read all available characters at once for paste handling
                    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if ready:
                        # Read all available characters immediately
                        import os

                        chars = []

                        # First read - get what's immediately available
                        try:
                            chunk = os.read(fd, 4096)
                            if chunk:
                                text_chunk = chunk.decode("utf-8", errors="replace")
                                chars.extend(list(text_chunk))
                        except (BlockingIOError, OSError, UnicodeDecodeError):
                            pass

                        # If we got at least one char, check for more (paste indicator)
                        if chars:
                            # Small delay to allow more characters to arrive (for paste)
                            time.sleep(0.01)

                            # Read additional characters that might have arrived
                            max_additional_reads = 5
                            for _ in range(max_additional_reads):
                                try:
                                    r, _, _ = select.select([sys.stdin], [], [], 0.001)
                                    if not r:
                                        break
                                    chunk = os.read(fd, 4096)
                                    if chunk:
                                        text_chunk = chunk.decode("utf-8", errors="replace")
                                        chars.extend(list(text_chunk))
                                    else:
                                        break
                                except (BlockingIOError, OSError, UnicodeDecodeError):
                                    break

                        # If no chars from os.read, try sys.stdin.read
                        if not chars:
                            try:
                                char = sys.stdin.read(1)
                                if char:
                                    chars = [char]
                            except Exception:
                                pass

                        if not chars:
                            continue

                        # Check if we have an escape sequence (arrow keys)
                        # Escape sequences come as: \x1b, [, A/B/C/D
                        if len(chars) >= 3 and chars[0] == "\x1b" and chars[1] == "[":
                            arrow_dir = chars[2]
                            if arrow_dir in ("A", "B", "C", "D"):
                                # This is an arrow key sequence
                                # Handle it specially if palette is active
                                if self.palette_active:
                                    filtered = self.command_registry.filter(self.palette_query)
                                    if arrow_dir == "A":  # Up arrow
                                        if filtered:
                                            self.palette_selected_index = (
                                                self.palette_selected_index - 1
                                            ) % len(filtered)
                                    elif arrow_dir == "B":  # Down arrow
                                        if filtered:
                                            self.palette_selected_index = (
                                                self.palette_selected_index + 1
                                            ) % len(filtered)
                                    # Skip other arrow directions (left/right) in palette mode
                                    if arrow_dir in ("A", "B"):
                                        continue
                                # For non-palette mode or left/right arrows, continue
                                # with normal processing. Remove the arrow sequence
                                if len(chars) > 3:
                                    chars = chars[3:]
                                else:
                                    continue

                        # If we got multiple chars, it's likely a paste
                        if len(chars) > 1:
                            paste_text = "".join(chars)

                            # Filter out control chars except newlines
                            filtered_chars = []
                            for c in paste_text:
                                if c in ("\r", "\n"):
                                    filtered_chars.append("\n" if c == "\r" else c)
                                elif len(c) == 1 and ord(c) >= 32:
                                    filtered_chars.append(c)

                            if filtered_chars:
                                text_to_insert = "".join(filtered_chars)
                                self.current_text = (
                                    self.current_text[: self.cursor_pos]
                                    + text_to_insert
                                    + self.current_text[self.cursor_pos :]
                                )
                                self.cursor_pos += len(text_to_insert)
                                # Force immediate panel update
                                live.update(
                                    self._create_input_panel(
                                        self.current_text, self.cursor_pos, self.cursor_visible
                                    )
                                )
                                continue

                        # Single character input
                        char = chars[0]
                    else:
                        continue

                    # Handle palette mode
                    if self.palette_active:
                        filtered = self.command_registry.filter(self.palette_query)
                        if not filtered:
                            filtered = []

                        # Handle Enter key - insert selected command
                        if char in ("\r", "\n"):
                            if filtered:
                                selected_cmd = filtered[self.palette_selected_index]
                                # Replace `/` + query with command template
                                # Query text: palette_start_pos+1 to cursor_pos
                                text_before = self.current_text[: self.palette_start_pos]
                                text_after = self.current_text[self.cursor_pos :]
                                self.current_text = text_before + selected_cmd.template + text_after
                                self.cursor_pos = self.palette_start_pos + len(
                                    selected_cmd.template
                                )
                                self.palette_active = False
                                self.palette_query = ""
                                continue
                            else:
                                # No matches, just exit palette
                                self.palette_active = False
                                self.palette_query = ""
                                continue
                        elif char == "\t":  # Tab - auto-complete first match
                            if filtered:
                                selected_cmd = filtered[0]
                                # Replace `/` + query with command template
                                text_before = self.current_text[: self.palette_start_pos]
                                text_after = self.current_text[self.cursor_pos :]
                                self.current_text = text_before + selected_cmd.template + text_after
                                self.cursor_pos = self.palette_start_pos + len(
                                    selected_cmd.template
                                )
                                self.palette_active = False
                                self.palette_query = ""
                                continue
                        elif char == "\x1b":  # Escape sequence
                            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                            if not ready:
                                # Just Escape - cancel palette
                                # Remove from palette_start_pos (the `/`) to cursor_pos
                                text_before = self.current_text[: self.palette_start_pos]
                                text_after = self.current_text[self.cursor_pos :]
                                self.current_text = text_before + text_after
                                self.cursor_pos = self.palette_start_pos
                                self.palette_active = False
                                self.palette_query = ""
                                continue

                            next1 = sys.stdin.read(1)
                            if not next1:
                                continue

                            # Handle arrow keys in palette mode
                            if next1 == "[":
                                ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                                next2 = sys.stdin.read(1) if ready else ""
                                if next2 == "A":  # Up arrow
                                    if filtered:
                                        self.palette_selected_index = (
                                            self.palette_selected_index - 1
                                        ) % len(filtered)
                                elif next2 == "B":  # Down arrow
                                    if filtered:
                                        self.palette_selected_index = (
                                            self.palette_selected_index + 1
                                        ) % len(filtered)
                            continue
                        elif char in ("\x7f", "\b"):  # Backspace in palette mode
                            if self.cursor_pos > self.palette_start_pos:
                                # Still have query characters to remove
                                self.palette_query = (
                                    self.palette_query[:-1] if len(self.palette_query) > 0 else ""
                                )
                                self.palette_selected_index = 0
                                # Remove from current_text
                                self.current_text = (
                                    self.current_text[: self.cursor_pos - 1]
                                    + self.current_text[self.cursor_pos :]
                                )
                                self.cursor_pos -= 1
                            else:
                                # Backspace at `/` position - exit palette and remove `/`
                                text_before = self.current_text[: self.palette_start_pos]
                                text_after = self.current_text[self.palette_start_pos + 1 :]
                                self.current_text = text_before + text_after
                                self.cursor_pos = self.palette_start_pos
                                self.palette_active = False
                                self.palette_query = ""
                            continue
                        elif (
                            len(char) == 1 and ord(char) >= 32
                        ):  # Printable characters in palette mode
                            self.palette_query += char
                            self.palette_selected_index = 0
                            # Also add to current_text
                            self.current_text = (
                                self.current_text[: self.cursor_pos]
                                + char
                                + self.current_text[self.cursor_pos :]
                            )
                            self.cursor_pos += 1
                            continue

                    # Handle Enter key - Enter submits
                    if char in ("\r", "\n"):
                        if not self.palette_active:  # Only submit if palette not active
                            break  # Submit on Enter
                    elif char == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    elif char == "\x04":  # Ctrl+D - submit
                        if self.current_text.strip():
                            break
                        else:
                            raise KeyboardInterrupt
                    elif char in ("\x7f", "\b"):  # Backspace
                        if self.cursor_pos > 0:
                            self.current_text = (
                                self.current_text[: self.cursor_pos - 1]
                                + self.current_text[self.cursor_pos :]
                            )
                            self.cursor_pos -= 1
                    elif char == "\x1b":  # Escape sequence (arrows, etc.)
                        # Read next character with timeout
                        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if not ready:
                            continue  # Just Escape key, ignore

                        next1 = sys.stdin.read(1)
                        if not next1:
                            continue

                        # Check for Alt+Enter (Esc+Enter) - inserts newline
                        if next1 in ("\r", "\n"):
                            # Alt+Enter inserts newline
                            self.current_text = (
                                self.current_text[: self.cursor_pos]
                                + "\n"
                                + self.current_text[self.cursor_pos :]
                            )
                            self.cursor_pos += 1
                            continue

                        # Read next character for arrow keys
                        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                        next2 = sys.stdin.read(1) if ready else ""

                        if next1 == "[":
                            if next2 == "D":  # Left arrow
                                if self.cursor_pos > 0:
                                    # Check if we're at the start of a line (after newline)
                                    if (
                                        self.cursor_pos > 0
                                        and self.current_text[self.cursor_pos - 1] == "\n"
                                    ):
                                        # Already at start of line, move to end of previous line
                                        line, col = self._get_cursor_position(
                                            self.current_text, self.cursor_pos
                                        )
                                        if line > 0:
                                            lines = self._get_text_lines(self.current_text)
                                            prev_line = lines[line - 1]
                                            self.cursor_pos = sum(
                                                len(line_text) + 1
                                                for line_text in lines[: line - 1]
                                            ) + len(prev_line)
                                    else:
                                        self.cursor_pos -= 1
                            elif next2 == "C":  # Right arrow
                                if self.cursor_pos < len(self.current_text):
                                    # Check if we're at the end of a line (before newline)
                                    if self.current_text[self.cursor_pos] == "\n":
                                        # Move to start of next line
                                        self.cursor_pos += 1
                                    else:
                                        self.cursor_pos += 1
                            elif next2 == "A":  # Up arrow
                                # Move cursor up one line
                                line, col = self._get_cursor_position(
                                    self.current_text, self.cursor_pos
                                )
                                if line > 0:
                                    lines = self._get_text_lines(self.current_text)
                                    prev_line = lines[line - 1]
                                    # Move to same column in previous line, or end if shorter
                                    target_col = min(col, len(prev_line))
                                    # Calculate new absolute position
                                    self.cursor_pos = (
                                        sum(len(line_text) + 1 for line_text in lines[: line - 1])
                                        + target_col
                                    )
                            elif next2 == "B":  # Down arrow
                                # Move cursor down one line
                                line, col = self._get_cursor_position(
                                    self.current_text, self.cursor_pos
                                )
                                lines = self._get_text_lines(self.current_text)
                                if line < len(lines) - 1:
                                    next_line = lines[line + 1]
                                    # Move to same column in next line, or end if shorter
                                    target_col = min(col, len(next_line))
                                    # Calculate new absolute position
                                    self.cursor_pos = (
                                        sum(len(line_text) + 1 for line_text in lines[: line + 1])
                                        + target_col
                                    )
                            elif next2 == "\r":  # Esc+Enter = submit
                                break
                    elif char == "\x0a" or char == "\x0d":  # Already handled above
                        pass
                    elif len(char) == 1 and ord(char) >= 32:  # Printable characters
                        # Check if `/` should activate palette
                        if char == "/" and self._should_activate_palette(
                            self.current_text, self.cursor_pos
                        ):
                            self.palette_active = True
                            self.palette_start_pos = self.cursor_pos
                            self.palette_query = ""
                            self.palette_selected_index = 0
                            # Insert `/` into text
                            self.current_text = (
                                self.current_text[: self.cursor_pos]
                                + char
                                + self.current_text[self.cursor_pos :]
                            )
                            self.cursor_pos += 1
                        else:
                            # Normal character input
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
            default_index if default_index is not None and 0 <= default_index < len(choices) else 0
        )

    def _create_select_panel(self, selected_idx: int) -> Panel:
        """Create selection panel with choices and highlighted selection.

        Implements scrolling window that follows the cursor.
        Window size is limited to a reasonable maximum (e.g., 15 items visible).
        """
        # Window configuration
        MAX_VISIBLE = 15
        total_choices = len(self.choices)

        # Calculate visible range with padding around selection
        # Try to keep selected item in middle of visible window
        window_size = min(MAX_VISIBLE, total_choices)
        half_window = window_size // 2

        # Calculate start index for visible window
        if total_choices <= window_size:
            # All items fit, show all
            start_idx = 0
            end_idx = total_choices
        else:
            # Calculate window to keep selected item visible
            start_idx = max(0, selected_idx - half_window)
            end_idx = min(total_choices, start_idx + window_size)

            # Adjust if we're at the end
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)

        visible_choices = self.choices[start_idx:end_idx]
        visible_start = start_idx

        content = Text()

        # Show indicator if there are items above visible window
        if start_idx > 0:
            content.append("  ...", style="dim")
            content.append(f" ({start_idx} more above)", style="dim")
            content.append("\n")

        for idx, choice in enumerate(visible_choices):
            actual_idx = visible_start + idx
            is_selected = actual_idx == selected_idx

            indicator = "❯ " if is_selected else "  "
            indicator_style = "bold cyan" if is_selected else "dim"
            content.append(indicator, style=indicator_style)

            try:
                choice_text = Text.from_markup(choice)
            except Exception:
                choice_text = Text(choice)

            if is_selected:
                highlighted = choice_text.copy()
                highlighted.stylize(Style(bold=True))
                highlighted.stylize(Style(bgcolor="grey11"), 0, len(highlighted))
                content.append_text(highlighted)
            else:
                content.append_text(choice_text)

            content.append("\n")

        # Show indicator if there are items below visible window
        if end_idx < total_choices:
            content.append("  ...", style="dim")
            content.append(f" ({total_choices - end_idx} more below)", style="dim")
            content.append("\n")

        # Add hint with subtle separator
        content.append("\n", style="")
        content.append("─" * 40, style="dim")
        content.append("\n")
        content.append("  ", style="")
        content.append("↑/↓", style="bold yellow")
        content.append(" Navigate  │  ", style="dim")
        content.append("Enter", style="bold green")
        content.append(" Select  │  ", style="dim")
        content.append("Ctrl+C", style="bold red")
        content.append(" Cancel", style="dim")

        # Show current position indicator
        if total_choices > window_size:
            content.append(f"  │  [{selected_idx + 1}/{total_choices}]", style="dim")

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
            return self.fallback_select()

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
                                self.selected_index = (self.selected_index - 1) % len(self.choices)
                            elif next2 == "B":  # Down arrow
                                self.selected_index = (self.selected_index + 1) % len(self.choices)

            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def fallback_select(self) -> str | None:
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
            return self.fallback_confirm()

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

    def fallback_confirm(self) -> bool | None:
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
    return selector.fallback_select()


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
    return confirmer.fallback_confirm()
