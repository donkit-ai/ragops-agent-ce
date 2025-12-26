"""Component protocols for UI abstraction.

Defines abstract interfaces for UI components like spinners,
progress bars, and live-updating regions.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Spinner(Protocol):
    """Loading spinner with optional message.

    Can be used as a context manager:
        with ui.create_spinner("Loading...") as spinner:
            # do work
            spinner.update("Still loading...")
    """

    def start(self) -> None:
        """Start the spinner animation."""
        ...

    def stop(self) -> None:
        """Stop the spinner animation."""
        ...

    def update(self, message: str) -> None:
        """Update the spinner message.

        Args:
            message: New message to display
        """
        ...

    def __enter__(self) -> Spinner:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        ...


@runtime_checkable
class ProgressBar(Protocol):
    """Progress bar with percentage/count display.

    Can be used as a context manager:
        with ui.create_progress(total=100) as progress:
            for i in range(100):
                progress.update(i, message="Processing...")
    """

    def update(self, current: int, message: str = "") -> None:
        """Update progress.

        Args:
            current: Current progress value
            message: Optional status message
        """
        ...

    def advance(self, amount: int = 1) -> None:
        """Advance progress by amount.

        Args:
            amount: Amount to advance (default 1)
        """
        ...

    def finish(self) -> None:
        """Mark progress as complete."""
        ...

    def __enter__(self) -> ProgressBar:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        ...


@runtime_checkable
class LiveContext(Protocol):
    """Live-updating display region.

    Used for streaming output that needs to be updated in place.

    Can be used as a context manager:
        with ui.create_live_context() as live:
            for chunk in stream:
                live.update(accumulated_content)
    """

    def update(self, content: str) -> None:
        """Update the displayed content.

        Args:
            content: New content to display
        """
        ...

    def refresh(self) -> None:
        """Force a display refresh."""
        ...

    def __enter__(self) -> LiveContext:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        ...
