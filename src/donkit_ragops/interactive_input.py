"""
Interactive input module for RAGOps Agent CE.

Provides interactive input box functionality with real-time typing inside Rich panels.
Follows Single Responsibility Principle - handles only user input interactions.
"""

import os
import sys

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PromptStyle

from donkit_ragops.command_palette import CommandRegistry
from donkit_ragops.ui import get_ui
from donkit_ragops.ui.styles import StyleName, styled_text


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

    # Track last printed status to avoid duplicates
    _last_status_hash: int = 0

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
            }
        )

        self.session = PromptSession(
            multiline=True,
            enable_history_search=True,
            key_bindings=bindings,
            style=style,
            completer=completer,
            complete_while_typing=True,
            complete_in_thread=False,
        )

    def get_input(self) -> str:
        """Get user input with prompt_toolkit."""
        try:
            if not sys.stdin.isatty():
                raise ImportError("Not running in a terminal")

            # Print visual input box with checklist status
            # Lazy import to avoid circular dependencies
            from donkit_ragops.checklist_manager import checklist_status_provider

            # Get terminal width for box
            try:
                terminal_width = os.get_terminal_size().columns
            except OSError:
                terminal_width = 80

            box_width = min(terminal_width - 1, 100)

            # Build input box using UI abstraction
            ui = get_ui()
            status = checklist_status_provider.status

            # Top border with optional status
            top_line = "╭" + "─" * (box_width - 2) + "╮"
            ui.print_styled(styled_text((StyleName.DIM, top_line)))

            # Status line if checklist is active
            if status.total > 0:
                # Update hash tracking
                status_tuple = (status.icon, status.description, status.completed, status.total)
                current_hash = hash(status_tuple)
                if current_hash != InteractiveInputBox._last_status_hash:
                    InteractiveInputBox._last_status_hash = current_hash

                # Determine icon style based on status
                if status.icon == "✓":
                    icon_style = StyleName.SUCCESS
                elif status.icon == "⚡":
                    icon_style = StyleName.WARNING
                else:
                    icon_style = StyleName.DIM

                progress = f"({status.completed}/{status.total})"
                ui.print_styled(
                    styled_text(
                        (StyleName.DIM, "│ "),
                        (icon_style, status.icon),
                        (None, f" {status.description} "),
                        (StyleName.DIM, progress),
                    )
                )

            # Bottom border connects to prompt
            ui.print("╰", StyleName.DIM, end="")

            # Use prompt_toolkit for input with styled prompt
            result = self.session.prompt(
                [("class:prompt", "❯ ")],
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
                    logger.debug(
                        f"Path expansion: result={result!r}, expanded={expanded_path!r}, "
                        f"exists={os.path.exists(expanded_path)}"
                    )
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
            ui = get_ui()
            ui.newline()
            ui.print_styled(styled_text((StyleName.BOLD, "> ")))
            try:
                return input().strip()
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
    input_box = InteractiveInputBox()
    return input_box.get_input()


def interactive_select(
    choices: list[str],
    title: str = "Select an option",
    default_index: int | None = None,
) -> str | None:
    """
    Show interactive selection menu with arrow key navigation.

    Delegates to UI abstraction layer.

    Args:
        choices: List of options to choose from
        title: Title for the selection menu
        default_index: Optional initial selection index

    Returns:
        Selected choice string or None if cancelled
    """
    ui = get_ui()
    return ui.select(choices, title, default_index or 0)


def interactive_confirm(question: str, default: bool = True) -> bool | None:
    """
    Show interactive yes/no confirmation with arrow key navigation.

    Delegates to UI abstraction layer.

    Args:
        question: Question to ask the user
        default: Default value (True for Yes, False for No)

    Returns:
        True for yes, False for no, None if cancelled
    """
    ui = get_ui()
    return ui.confirm(question, default)
