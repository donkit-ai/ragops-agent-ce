"""Rich UI adapter implementation.

Provides full-featured UI using the Rich library for terminal output.
"""

from __future__ import annotations

import select as select_module
import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.status import Status
from rich.style import Style
from rich.text import Text

from donkit_ragops.ui.components import LiveContext, ProgressBar, Spinner
from donkit_ragops.ui.styles import RICH_STYLES, StyledText, StyleName

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

INTERACTIVE_AVAILABLE = TERMIOS_AVAILABLE or MSVCRT_AVAILABLE


def _read_key_windows() -> str:
    """Read a key from Windows console."""
    if not MSVCRT_AVAILABLE:
        raise ImportError("msvcrt not available")

    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            ch2 = msvcrt.getch()
            if ch2 == b"H":
                return "\x1b[A"
            elif ch2 == b"P":
                return "\x1b[B"
            elif ch2 == b"M":
                return "\x1b[C"
            elif ch2 == b"K":
                return "\x1b[D"
            return ""
        return ch.decode("utf-8", errors="ignore")
    return ""


def _read_key_unix() -> str:
    """Read a key from Unix terminal."""
    if not TERMIOS_AVAILABLE:
        raise ImportError("termios not available")

    if sys.stdin in select_module.select([sys.stdin], [], [], 0.05)[0]:
        return sys.stdin.read(1)
    return ""


class RichSpinner:
    """Rich Status-based spinner implementation."""

    def __init__(self, console: Console, message: str):
        self._console = console
        self._message = message
        self._status: Status | None = None

    def start(self) -> None:
        """Start the spinner."""
        self._status = Status(self._message, spinner="dots", console=self._console)
        self._status.start()

    def stop(self) -> None:
        """Stop the spinner."""
        if self._status:
            self._status.stop()
            self._status = None

    def update(self, message: str) -> None:
        """Update spinner message."""
        self._message = message
        if self._status:
            self._status.update(message)

    def __enter__(self) -> RichSpinner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class RichProgressBar:
    """Rich Progress-based progress bar implementation."""

    def __init__(self, console: Console, total: int, description: str):
        self._console = console
        self._total = total
        self._description = description
        self._current = 0
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def start(self) -> None:
        """Start the progress bar."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self._console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self._description, total=self._total)

    def update(self, current: int, message: str = "") -> None:
        """Update progress."""
        self._current = current
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=current,
                description=message or self._description,
            )

    def advance(self, amount: int = 1) -> None:
        """Advance progress."""
        self._current += amount
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=amount)

    def finish(self) -> None:
        """Complete the progress."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=self._total)

    def stop(self) -> None:
        """Stop the progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def __enter__(self) -> RichProgressBar:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class RichLiveContext:
    """Rich Live-based live update context."""

    def __init__(self, console: Console):
        self._console = console
        self._live: Live | None = None
        self._content = ""

    def start(self) -> None:
        """Start live updates."""
        self._live = Live(Text(""), console=self._console, refresh_per_second=20)
        self._live.start()

    def update(self, content: str) -> None:
        """Update displayed content."""
        self._content = content
        if self._live:
            self._live.update(Text(content))

    def refresh(self) -> None:
        """Force refresh."""
        if self._live:
            self._live.refresh()

    def stop(self) -> None:
        """Stop live updates."""
        if self._live:
            self._live.stop()
            self._live = None

    def __enter__(self) -> RichLiveContext:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class RichUI:
    """Rich-based UI implementation.

    Provides full-featured terminal UI using the Rich library.
    """

    def __init__(self) -> None:
        self._console = Console()

    @property
    def console(self) -> Console:
        """Get the underlying Rich console (for advanced usage)."""
        return self._console

    def _apply_style(self, message: str, style: StyleName | None) -> str:
        """Apply Rich markup for a style."""
        if style is None:
            return message
        rich_style = RICH_STYLES.get(style, "")
        if rich_style:
            return f"[{rich_style}]{message}[/{rich_style}]"
        return message

    # === OUTPUT ===

    def print(self, message: str, style: StyleName | None = None, end: str = "\n") -> None:
        """Print a message with optional styling."""
        styled = self._apply_style(message, style)
        self._console.print(styled, end=end)

    def print_styled(self, styled_text: StyledText) -> None:
        """Print styled text segments."""
        text = Text()
        for style, content in styled_text:
            if style is None:
                text.append(content)
            else:
                rich_style = RICH_STYLES.get(style, "")
                text.append(content, style=rich_style)
        self._console.print(text)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self._console.print(f"[bold red]Error:[/bold red] {message}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self._console.print(f"[bold green]✓[/bold green] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self._console.print(f"[bold yellow]⚠[/bold yellow] {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self._console.print(f"[bold blue]ℹ[/bold blue] {message}")

    def print_markdown(self, content: str) -> None:
        """Print markdown-formatted content."""
        md = Markdown(content)
        self._console.print(md)

    def print_panel(
        self,
        content: str,
        title: str = "",
        border_style: StyleName | None = None,
    ) -> None:
        """Print content in a bordered panel."""
        rich_border = RICH_STYLES.get(border_style, "cyan") if border_style else "cyan"
        panel = Panel(
            content,
            title=f"[bold]{title}[/bold]" if title else None,
            border_style=rich_border,
        )
        self._console.print(panel)

    def newline(self) -> None:
        """Print a blank line."""
        self._console.print()

    def clear(self) -> None:
        """Clear the terminal screen."""
        self._console.clear()

    # === PROGRESS COMPONENTS ===

    def create_spinner(self, message: str = "Loading...") -> Spinner:
        """Create a loading spinner."""
        return RichSpinner(self._console, message)

    def create_progress(self, total: int = 100, description: str = "") -> ProgressBar:
        """Create a progress bar."""
        return RichProgressBar(self._console, total, description)

    # === INTERACTIVE COMPONENTS ===

    def select(
        self,
        choices: list[str],
        title: str = "Select an option",
        default_index: int = 0,
    ) -> str | None:
        """Show interactive selection menu with arrow key navigation."""
        if not INTERACTIVE_AVAILABLE or not sys.stdin.isatty():
            return self._fallback_select(choices, title, default_index)

        try:
            return self._interactive_select(choices, title, default_index)
        except (ImportError, OSError):
            return self._fallback_select(choices, title, default_index)

    def _interactive_select(
        self,
        choices: list[str],
        title: str,
        default_index: int,
    ) -> str | None:
        """Interactive selection with arrow keys."""
        selected_index = default_index if 0 <= default_index < len(choices) else 0

        old_settings = None
        if TERMIOS_AVAILABLE:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        with Live(
            self._create_select_panel(choices, title, selected_index),
            console=self._console,
            refresh_per_second=20,
        ) as live:
            try:
                while True:
                    live.update(self._create_select_panel(choices, title, selected_index))

                    if MSVCRT_AVAILABLE:
                        char = _read_key_windows()
                        if not char:
                            time.sleep(0.05)
                            continue
                    else:
                        char = _read_key_unix()
                        if not char:
                            continue

                    if char in ("\r", "\n"):
                        return choices[selected_index]
                    elif char == "\x03":  # Ctrl+C
                        return None
                    elif char == "\x04":  # Ctrl+D
                        return None
                    elif char == "\x1b[A":  # Up
                        selected_index = (selected_index - 1) % len(choices)
                    elif char == "\x1b[B":  # Down
                        selected_index = (selected_index + 1) % len(choices)
                    elif char == "\x1b" and not MSVCRT_AVAILABLE:
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == "[":
                            if next2 == "A":
                                selected_index = (selected_index - 1) % len(choices)
                            elif next2 == "B":
                                selected_index = (selected_index + 1) % len(choices)
            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _create_select_panel(self, choices: list[str], title: str, selected_idx: int) -> Panel:
        """Create selection panel with choices."""
        MAX_VISIBLE = 15
        total_choices = len(choices)
        window_size = min(MAX_VISIBLE, total_choices)
        half_window = window_size // 2

        if total_choices <= window_size:
            start_idx = 0
            end_idx = total_choices
        else:
            start_idx = max(0, selected_idx - half_window)
            end_idx = min(total_choices, start_idx + window_size)
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)

        content = Text()

        if start_idx > 0:
            content.append(f"  ... ({start_idx} more above)\n", style="dim")

        for idx, choice in enumerate(choices[start_idx:end_idx]):
            actual_idx = start_idx + idx
            is_selected = actual_idx == selected_idx

            indicator = "❯ " if is_selected else "  "
            content.append(indicator, style="bold cyan" if is_selected else "dim")

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

        if end_idx < total_choices:
            content.append(f"  ... ({total_choices - end_idx} more below)\n", style="dim")

        content.append("\n")
        content.append("─" * 40, style="dim")
        content.append("\n  ")
        content.append("↑/↓", style="bold yellow")
        content.append(" Navigate  │  ", style="dim")
        content.append("Enter", style="bold green")
        content.append(" Select  │  ", style="dim")
        content.append("Ctrl+C", style="bold red")
        content.append(" Cancel", style="dim")

        if total_choices > window_size:
            content.append(f"  │  [{selected_idx + 1}/{total_choices}]", style="dim")

        return Panel(
            content,
            title=f"[bold cyan]{title}[/bold cyan]",
            title_align="left",
            border_style="cyan",
            expand=True,
            padding=(1, 2),
        )

    def _fallback_select(self, choices: list[str], title: str, default_index: int) -> str | None:
        """Fallback to numbered selection."""
        self.newline()
        self._console.print(f"[bold]{title}[/bold]")
        for i, choice in enumerate(choices, 1):
            try:
                choice_text = Text.from_markup(choice).plain
            except Exception:
                choice_text = choice
            self._console.print(f"  {i}. {choice_text}")
        self.newline()

        while True:
            try:
                response = input(f"Enter number (1-{len(choices)}) or 'q' to cancel: ").strip()
                if response.lower() in ("q", "quit", "cancel"):
                    return None
                if not response:
                    return choices[default_index]
                num = int(response)
                if 1 <= num <= len(choices):
                    return choices[num - 1]
                self.print_error(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                self.print_error("Please enter a valid number")
            except (EOFError, KeyboardInterrupt):
                return None

    def confirm(self, question: str, default: bool = True) -> bool | None:
        """Show yes/no confirmation dialog with arrow key navigation."""
        if not INTERACTIVE_AVAILABLE or not sys.stdin.isatty():
            return self._fallback_confirm(question, default)

        try:
            return self._interactive_confirm(question, default)
        except (ImportError, OSError):
            return self._fallback_confirm(question, default)

    def _interactive_confirm(self, question: str, default: bool) -> bool | None:
        """Interactive confirmation with arrow keys."""
        selected_yes = default

        old_settings = None
        if TERMIOS_AVAILABLE:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        with Live(
            self._create_confirm_panel(question, selected_yes),
            console=self._console,
            refresh_per_second=20,
        ) as live:
            try:
                while True:
                    live.update(self._create_confirm_panel(question, selected_yes))

                    if MSVCRT_AVAILABLE:
                        char = _read_key_windows()
                        if not char:
                            time.sleep(0.05)
                            continue
                    else:
                        char = _read_key_unix()
                        if not char:
                            continue

                    if char in ("\r", "\n"):
                        return selected_yes
                    elif char in ("y", "Y"):
                        return True
                    elif char in ("n", "N"):
                        return False
                    elif char == "\x03":  # Ctrl+C
                        return None
                    elif char == "\x04":  # Ctrl+D
                        return None
                    elif char in ("\x1b[C", "\x1b[D"):  # Right or Left
                        selected_yes = not selected_yes
                    elif char == "\x1b" and not MSVCRT_AVAILABLE:
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == "[" and next2 in ("C", "D"):
                            selected_yes = not selected_yes
            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _create_confirm_panel(self, question: str, selected_yes: bool) -> Panel:
        """Create confirmation panel."""
        content = Text()
        content.append(question, style="white")
        content.append("\n\n")

        if selected_yes:
            content.append("❯ ", style="bold green")
            content.append("Yes", style="bold green on black")
        else:
            content.append("  ", style="dim")
            content.append("Yes", style="white")

        content.append("  ")

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

    def _fallback_confirm(self, question: str, default: bool) -> bool | None:
        """Fallback to y/n prompt."""
        default_str = "Y/n" if default else "y/N"
        try:
            response = input(f"{question} [{default_str}]: ").strip().lower()
            if not response:
                return default
            if response in ("y", "yes"):
                return True
            if response in ("n", "no"):
                return False
            return default
        except (EOFError, KeyboardInterrupt):
            return None

    def text_input(self, prompt: str = "> ") -> str:
        """Get text input from user with interactive input box."""
        from donkit_ragops.interactive_input import get_user_input

        return get_user_input()

    # === LIVE UPDATES ===

    def create_live_context(self) -> LiveContext:
        """Create a live-updating display region."""
        return RichLiveContext(self._console)
