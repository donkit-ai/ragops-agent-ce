"""Prompt Toolkit UI adapter implementation.

Provides UI using prompt_toolkit for terminal output.
Designed for use with async REPL loops where prompt_toolkit manages input.
"""

from __future__ import annotations

import sys
from typing import Any

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style

from donkit_ragops.ui.components import LiveContext, ProgressBar, Spinner
from donkit_ragops.ui.styles import StyledText, StyleName

# Map our style names to prompt_toolkit style classes
PT_STYLES = Style.from_dict(
    {
        "error": "bold #ff0000",
        "success": "bold #00ff00",
        "warning": "bold #ffff00",
        "info": "bold #00ffff",
        "dim": "#888888",
        "bold": "bold",
        "agent-prefix": "bold #00ff00",
        "tool-name": "#ffff00",
        "tool-args": "#888888",
        "code": "#00ffff",
        "header": "bold #ff00ff",
    }
)

# Map StyleName enum to prompt_toolkit style class names
STYLE_MAP: dict[StyleName, str] = {
    StyleName.ERROR: "class:error",
    StyleName.SUCCESS: "class:success",
    StyleName.WARNING: "class:warning",
    StyleName.INFO: "class:info",
    StyleName.DIM: "class:dim",
    StyleName.BOLD: "class:bold",
    StyleName.AGENT_PREFIX: "class:agent-prefix",
    StyleName.USER_PREFIX: "class:bold",
    StyleName.TOOL_NAME: "class:tool-name",
    StyleName.TOOL_ARGS: "class:tool-args",
    StyleName.CODE: "class:code",
    StyleName.PATH: "class:info",
    StyleName.URL: "class:info",
}


class PromptToolkitSpinner:
    """Simple text-based spinner for prompt_toolkit."""

    def __init__(self, message: str):
        self._message = message
        self._running = False

    def start(self) -> None:
        """Start the spinner (just prints message)."""
        self._running = True
        print_formatted_text(
            FormattedText([("class:dim", f"⠋ {self._message}")]),
            style=PT_STYLES,
        )

    def stop(self) -> None:
        """Stop the spinner."""
        self._running = False

    def update(self, message: str) -> None:
        """Update spinner message."""
        self._message = message
        if self._running:
            print_formatted_text(
                FormattedText([("class:dim", f"⠋ {self._message}")]),
                style=PT_STYLES,
            )

    def __enter__(self) -> PromptToolkitSpinner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class PromptToolkitProgressBar:
    """Simple text-based progress bar for prompt_toolkit."""

    def __init__(self, total: int, description: str):
        self._total = total
        self._description = description
        self._current = 0

    def start(self) -> None:
        """Start the progress bar."""
        self._print_progress()

    def _print_progress(self) -> None:
        """Print current progress."""
        if self._total > 0:
            percentage = (self._current / self._total) * 100
            bar_width = 20
            filled = int(bar_width * self._current / self._total)
            bar = "█" * filled + "░" * (bar_width - filled)
            print_formatted_text(
                FormattedText([("class:dim", f"{self._description} [{bar}] {percentage:.0f}%")]),
                style=PT_STYLES,
            )

    def update(self, current: int, message: str = "") -> None:
        """Update progress."""
        self._current = current
        if message:
            self._description = message
        self._print_progress()

    def advance(self, amount: int = 1) -> None:
        """Advance progress."""
        self._current += amount
        self._print_progress()

    def finish(self) -> None:
        """Complete the progress."""
        self._current = self._total
        self._print_progress()

    def stop(self) -> None:
        """Stop the progress bar."""
        pass

    def __enter__(self) -> PromptToolkitProgressBar:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class PromptToolkitLiveContext:
    """Simple live context for prompt_toolkit (just prints updates)."""

    def __init__(self) -> None:
        self._content = ""

    def start(self) -> None:
        """Start live updates."""
        pass

    def update(self, content: str) -> None:
        """Update displayed content."""
        self._content = content
        print_formatted_text(content)

    def refresh(self) -> None:
        """Force refresh."""
        pass

    def stop(self) -> None:
        """Stop live updates."""
        pass

    def __enter__(self) -> PromptToolkitLiveContext:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class PromptToolkitUI:
    """Prompt Toolkit-based UI implementation.

    Uses prompt_toolkit for all output, avoiding conflicts with
    prompt_toolkit input handling.
    """

    def __init__(self) -> None:
        self._output = create_output()

    def _style_class(self, style: StyleName | None) -> str:
        """Get prompt_toolkit style class for a StyleName."""
        if style is None:
            return ""
        return STYLE_MAP.get(style, "")

    # === OUTPUT ===

    def print(self, message: str, style: StyleName | None = None, end: str = "\n") -> None:
        """Print a message with optional styling."""
        style_class = self._style_class(style)
        if style_class:
            print_formatted_text(
                FormattedText([(style_class, message)]),
                style=PT_STYLES,
                end=end,
            )
        else:
            print_formatted_text(message, end=end)

    def print_styled(self, styled_text: StyledText) -> None:
        """Print styled text segments."""
        formatted_parts: list[tuple[str, str]] = []
        for style, content in styled_text:
            style_class = self._style_class(style) if style else ""
            formatted_parts.append((style_class, content))
        print_formatted_text(FormattedText(formatted_parts), style=PT_STYLES)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        print_formatted_text(
            FormattedText([("class:error", "Error: "), ("", message)]),
            style=PT_STYLES,
        )

    def print_success(self, message: str) -> None:
        """Print a success message."""
        print_formatted_text(
            FormattedText([("class:success", "✓ "), ("", message)]),
            style=PT_STYLES,
        )

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        print_formatted_text(
            FormattedText([("class:warning", "⚠ "), ("", message)]),
            style=PT_STYLES,
        )

    def print_info(self, message: str) -> None:
        """Print an info message."""
        print_formatted_text(
            FormattedText([("class:info", "ℹ "), ("", message)]),
            style=PT_STYLES,
        )

    def print_markdown(self, content: str) -> None:
        """Print markdown-formatted content.

        Basic markdown rendering for prompt_toolkit.
        """
        # Simple markdown: bold **text**, italic *text*, code `text`
        import re

        result = content
        # Bold
        result = re.sub(r"\*\*(.+?)\*\*", r"\033[1m\1\033[0m", result)
        # Code
        result = re.sub(r"`([^`]+)`", r"\033[36m\1\033[0m", result)
        print_formatted_text(result)

    def print_panel(
        self,
        content: str,
        title: str = "",
        border_style: StyleName | None = None,
    ) -> None:
        """Print content in a simple bordered panel."""
        width = 60
        border = "─" * width

        if title:
            print_formatted_text(
                FormattedText([("class:bold", f"┌─ {title} "), ("class:dim", border)])
            )
        else:
            print_formatted_text(FormattedText([("class:dim", f"┌{border}")]))

        for line in content.split("\n"):
            print_formatted_text(f"│ {line}")

        print_formatted_text(FormattedText([("class:dim", f"└{border}")]))

    def newline(self) -> None:
        """Print a blank line."""
        print_formatted_text("")

    def clear(self) -> None:
        """Clear the terminal screen."""
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")

    # === PROGRESS COMPONENTS ===

    def create_spinner(self, message: str = "Loading...") -> Spinner:
        """Create a loading spinner."""
        return PromptToolkitSpinner(message)

    def create_progress(self, total: int = 100, description: str = "") -> ProgressBar:
        """Create a progress bar."""
        return PromptToolkitProgressBar(total, description)

    # === INTERACTIVE COMPONENTS ===

    def select(
        self,
        choices: list[str],
        title: str = "Select an option",
        default_index: int = 0,
    ) -> str | None:
        """Show interactive selection menu."""
        from donkit_ragops.interactive_input import interactive_select

        return interactive_select(choices, title, default_index)

    def confirm(self, question: str, default: bool = True) -> bool | None:
        """Show yes/no confirmation dialog."""
        from donkit_ragops.interactive_input import interactive_confirm

        return interactive_confirm(question, default)

    def text_input(self, prompt: str = "> ") -> str:
        """Get text input from user."""
        from donkit_ragops.interactive_input import get_user_input

        return get_user_input()

    # === LIVE UPDATES ===

    def create_live_context(self) -> LiveContext:
        """Create a live-updating display region."""
        return PromptToolkitLiveContext()
