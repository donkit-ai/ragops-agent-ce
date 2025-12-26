"""UI protocol definition.

Defines the abstract interface for UI implementations.
Uses Python's Protocol for structural subtyping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from donkit_ragops.ui.components import LiveContext, ProgressBar, Spinner
    from donkit_ragops.ui.styles import StyledText, StyleName


@runtime_checkable
class UI(Protocol):
    """Abstract UI interface - output + interactive components.

    This protocol defines all UI operations that can be performed.
    Implementations (RichUI, PlainUI, etc.) provide concrete behavior.
    """

    # === OUTPUT ===

    def print(self, message: str, style: StyleName | None = None, *, end: str = "\n") -> None:
        """Print a message with optional styling.

        Args:
            message: Text to print
            style: Optional semantic style
            end: String to print at the end (default newline)
        """
        ...

    def print_styled(self, styled_text: StyledText) -> None:
        """Print styled text segments.

        Args:
            styled_text: List of (style, text) tuples
        """
        ...

    def print_error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: Error message
        """
        ...

    def print_success(self, message: str) -> None:
        """Print a success message.

        Args:
            message: Success message
        """
        ...

    def print_warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: Warning message
        """
        ...

    def print_info(self, message: str) -> None:
        """Print an info message.

        Args:
            message: Info message
        """
        ...

    def print_markdown(self, content: str) -> None:
        """Print markdown-formatted content.

        Args:
            content: Markdown string
        """
        ...

    def print_panel(
        self,
        content: str,
        title: str = "",
        border_style: StyleName | None = None,
    ) -> None:
        """Print content in a bordered panel.

        Args:
            content: Panel content
            title: Optional panel title
            border_style: Optional border style
        """
        ...

    def newline(self) -> None:
        """Print a blank line."""
        ...

    # === PROGRESS COMPONENTS ===

    def create_spinner(self, message: str = "Loading...") -> Spinner:
        """Create a loading spinner.

        Args:
            message: Spinner message

        Returns:
            Spinner instance (context manager)
        """
        ...

    def create_progress(self, total: int = 100, description: str = "") -> ProgressBar:
        """Create a progress bar.

        Args:
            total: Total steps
            description: Progress description

        Returns:
            ProgressBar instance (context manager)
        """
        ...

    # === INTERACTIVE COMPONENTS ===

    def select(
        self,
        choices: list[str],
        title: str = "Select an option",
        default_index: int = 0,
    ) -> str | None:
        """Show interactive selection menu.

        Args:
            choices: List of options
            title: Menu title
            default_index: Initially selected index

        Returns:
            Selected choice or None if cancelled
        """
        ...

    def confirm(self, question: str, default: bool = True) -> bool | None:
        """Show yes/no confirmation dialog.

        Args:
            question: Question to ask
            default: Default value (True=Yes, False=No)

        Returns:
            True for yes, False for no, None if cancelled
        """
        ...

    def text_input(self, prompt: str = "> ") -> str:
        """Get text input from user.

        Args:
            prompt: Input prompt

        Returns:
            User input string

        Raises:
            KeyboardInterrupt: If user cancels
        """
        ...

    # === LIVE UPDATES ===

    def create_live_context(self) -> LiveContext:
        """Create a live-updating display region.

        Returns:
            LiveContext instance (context manager)
        """
        ...
