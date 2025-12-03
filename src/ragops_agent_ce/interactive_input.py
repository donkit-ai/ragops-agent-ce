"""
Interactive input module for RAGOps Agent CE.

Provides interactive input box functionality with real-time typing inside Rich panels.
Follows Single Responsibility Principle - handles only user input interactions.
"""

import os
import select
import sys
import time
from typing import TYPE_CHECKING

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PromptStyle
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from ragops_agent_ce.command_palette import CommandRegistry

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


class CommandCompleter(Completer):
    """Command palette completer for prompt_toolkit."""

    def __init__(self, registry: CommandRegistry):
        self.registry = registry
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        """Get completions for commands and file paths."""
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        # 1. Command Completion (Start of line starting with '/')
        if text.startswith("/"):
            query = text[1:].lstrip()

            # Yield commands matching the query
            has_commands = False
            for cmd in self.registry.filter(query):
                has_commands = True
                display_meta = f"[{cmd.category}] {cmd.description}"
                yield Completion(
                    cmd.template,
                    start_position=-len(text),
                    display=cmd.name,
                    display_meta=display_meta,
                    style="class:command",
                )

            # 2. Absolute Path Completion (fallback if no commands)
            # Only show if no commands match and user is typing a path-like structure
            if len(text) > 1 and (not has_commands or "/" in query):
                yield from self.path_completer.get_completions(document, complete_event)

        # 3. File Navigation via '@' trigger
        # If word starts with @, trigger path completion
        elif word.startswith("@"):
            # Create a temp document without the '@' prefix for the path completer
            from prompt_toolkit.document import Document

            # content after @
            path_query = word[1:]

            # We need to get completions for 'path_text'
            # Since PathCompleter takes (document, event), we need a proxy document

            # Reconstruct document line without the leading @ of the current word
            # This is complex because we need to find where the word starts
            word_start_pos = text.rfind(word)
            if word_start_pos == -1:
                word_start_pos = 0

            # New text: ... [path_text] ...
            # effectively removing @ from the current position

            # Use get_completions with a modified document
            new_text_before_cursor = text[:word_start_pos] + path_query
            new_doc = Document(
                text=new_text_before_cursor + document.text_after_cursor,
                cursor_position=len(new_text_before_cursor),
            )

            for completion in self.path_completer.get_completions(new_doc, complete_event):
                # We need to adjust the completion to replace the @ as well
                # completion.start_position is relative to cursor in new_doc

                yield Completion(
                    completion.text,
                    start_position=completion.start_position,
                    display=completion.display,
                    display_meta=completion.display_meta,
                    style=completion.style,
                )

        # 4. Absolute Path Completion (Arguments - anywhere in line)
        elif word.startswith(("/", ".", "~")):
            yield from self.path_completer.get_completions(document, complete_event)


class InteractiveInputBox:
    """Handles interactive input using prompt_toolkit for better cross-platform support."""

    def __init__(self):
        self.command_registry = CommandRegistry()

        # Create completer for commands
        completer = CommandCompleter(self.command_registry)

        # Custom key bindings: Enter submits, Meta+Enter inserts newline
        bindings = KeyBindings()

        @bindings.add("enter", filter=Condition(lambda: True))
        def _(event):
            """Handle Enter: Drill down directories if completing, otherwise submit."""
            buffer = event.current_buffer
            # Check if completion menu is active and we have a selection
            if buffer.complete_state and buffer.complete_state.current_completion:
                completion = buffer.complete_state.current_completion
                text = completion.text

                # Apply the completion
                buffer.apply_completion(completion)

                # If it looks like a directory (ends with path separator), restart completion
                if text.endswith("/") or text.endswith("\\"):
                    buffer.start_completion()
            else:
                # No completion selected, submit the buffer
                buffer.validate_and_handle()

        @bindings.add("c-space")
        def _(event):
            """Force submit input on Ctrl+Space."""
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def _(event):
            """Insert newline on Meta+Enter."""
            event.current_buffer.insert_text("\n")

        @bindings.add("/")
        def _(event):
            """Insert '/' and trigger command completion."""
            buffer = event.current_buffer
            buffer.insert_text("/")
            buffer.start_completion()

        @bindings.add("@")
        def _(event):
            """Insert '@' and trigger file completion."""
            buffer = event.current_buffer
            buffer.insert_text("@")
            buffer.start_completion()

        @bindings.add("escape")
        def _(event):
            """Handle Esc key - cancel completion or clear buffer."""
            buffer = event.current_buffer

            # If completion menu is open, close it
            if buffer.complete_state:
                buffer.cancel_completion()
            # If there's text in buffer, clear it
            elif buffer.text:
                buffer.reset()
            # If buffer is empty, do nothing (use Ctrl+C to stop agent)

        # Custom style for the prompt
        style = PromptStyle.from_dict(
            {
                "prompt": "bold ansiblue",
                "": "#ffffff",  # Default text color
                # Completion menu styling
                "completion-menu": "bg:#232323 #ffffff",
                "completion-menu.completion": "bg:#232323 #ffffff",
                "completion-menu.completion.current": "bg:#444444 #ffffff bold",
                "completion-menu.meta.completion": "bg:#232323 #aaaaaa",
                "completion-menu.meta.completion.current": "bg:#444444 #aaaaaa",
                "bottom-toolbar": "bg:#232323",
                "bottom-toolbar.key": "bg:#232323 bold",
            }
        )

        def get_toolbar():
            return [
                ("class:bottom-toolbar", " "),
                ("class:bottom-toolbar.key", "/"),
                ("class:bottom-toolbar", " Commands "),
                ("class:bottom-toolbar.key", "@"),
                ("class:bottom-toolbar", " Files "),
                ("class:bottom-toolbar.key", "Tab"),
                ("class:bottom-toolbar", " Complete "),
                ("class:bottom-toolbar.key", "Meta(ctrl/cmd)+Enter"),
                ("class:bottom-toolbar", " Newline "),
                ("class:bottom-toolbar.key", ":q/quite/exit"),
                ("class:bottom-toolbar", " Exit "),
            ]

        self.session = PromptSession(
            multiline=True,
            enable_history_search=True,
            key_bindings=bindings,
            style=style,
            completer=completer,
            complete_while_typing=True,
            complete_in_thread=False,
            bottom_toolbar=get_toolbar,
        )

    def get_input(self) -> str:
        """Get user input with prompt_toolkit."""
        try:
            if not sys.stdin.isatty():
                raise ImportError("Not running in a terminal")

            # Use prompt_toolkit for input with styled prompt
            result = self.session.prompt(
                [("class:prompt", "you> ")],
                in_thread=True,
            )

            # Check if result is a path and expand it
            if result and (
                result.startswith("~")
                or result.startswith("/")
                or result.startswith(".")
                or result.startswith("@")
            ):
                # If starts with @, remove it before expansion
                is_at_trigger = result.startswith("@")
                path_to_check = result[1:] if is_at_trigger else result

                try:
                    expanded_path = os.path.abspath(os.path.expanduser(path_to_check))
                    if is_at_trigger or os.path.exists(expanded_path):
                        return expanded_path
                except Exception as ex:
                    logger.warning(f"Failed to expand path '{result}': {ex}")
                    # For @ trigger, at least return the path without @
                    if is_at_trigger:
                        return path_to_check
            return result
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt
        except (ImportError, OSError):
            # Fallback to simple input
            console.print()
            console.print("[bold blue]you>[/bold blue] ", end="")
            try:
                return input().strip()
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
    input_box = InteractiveInputBox()
    return input_box.get_input()


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
