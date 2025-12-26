"""Plain text UI adapter implementation.

Provides simple text output for testing, CI/CD, and piped output.
No colors, no interactive features - just plain text.
"""

from __future__ import annotations

from typing import Any

from donkit_ragops.ui.styles import StyledText, StyleName


class PlainSpinner:
    """Plain text spinner (just prints messages)."""

    def __init__(self, message: str):
        self._message = message

    def start(self) -> None:
        """Start the 'spinner' (just print message)."""
        print(f"... {self._message}")

    def stop(self) -> None:
        """Stop the 'spinner'."""
        pass

    def update(self, message: str) -> None:
        """Update message."""
        self._message = message
        print(f"... {message}")

    def __enter__(self) -> PlainSpinner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class PlainProgressBar:
    """Plain text progress bar."""

    def __init__(self, total: int, description: str):
        self._total = total
        self._description = description
        self._current = 0

    def start(self) -> None:
        """Start progress."""
        print(f"[0/{self._total}] {self._description}")

    def update(self, current: int, message: str = "") -> None:
        """Update progress."""
        self._current = current
        desc = message or self._description
        percent = (current / self._total * 100) if self._total > 0 else 0
        print(f"[{current}/{self._total}] ({percent:.0f}%) {desc}")

    def advance(self, amount: int = 1) -> None:
        """Advance progress."""
        self._current += amount
        self.update(self._current)

    def finish(self) -> None:
        """Complete progress."""
        self._current = self._total
        print(f"[{self._total}/{self._total}] (100%) Done")

    def stop(self) -> None:
        """Stop progress."""
        pass

    def __enter__(self) -> PlainProgressBar:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class PlainLiveContext:
    """Plain text live context (just prints updates)."""

    def __init__(self) -> None:
        self._content = ""

    def start(self) -> None:
        """Start live context."""
        pass

    def update(self, content: str) -> None:
        """Update content (prints new content)."""
        # Only print if content changed
        if content != self._content:
            # Print only the new part
            if content.startswith(self._content):
                new_part = content[len(self._content) :]
                if new_part:
                    print(new_part, end="", flush=True)
            else:
                print(content, end="", flush=True)
            self._content = content

    def refresh(self) -> None:
        """Refresh (no-op for plain text)."""
        pass

    def stop(self) -> None:
        """Stop live context."""
        print()  # Final newline

    def __enter__(self) -> PlainLiveContext:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()


class PlainUI:
    """Plain text UI implementation.

    Simple text output without colors or interactive features.
    Useful for testing, CI/CD, and piped output.
    """

    # === OUTPUT ===

    def print(self, message: str, style: StyleName | None = None, end: str = "\n") -> None:
        """Print a message (style is ignored)."""
        print(message, end=end)

    def print_styled(self, styled_text: StyledText) -> None:
        """Print styled text (styles are ignored)."""
        text = "".join(content for _, content in styled_text)
        print(text)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        print(f"Error: {message}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        print(f"✓ {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        print(f"Warning: {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        print(f"Info: {message}")

    def print_markdown(self, content: str) -> None:
        """Print markdown as plain text."""
        print(content)

    def print_panel(
        self,
        content: str,
        title: str = "",
        border_style: StyleName | None = None,
    ) -> None:
        """Print content in a simple text 'panel'."""
        if title:
            print(f"=== {title} ===")
        print(content)
        if title:
            print("=" * (len(title) + 8))

    def newline(self) -> None:
        """Print a blank line."""
        print()

    def clear(self) -> None:
        """Clear the screen (print some newlines in plain mode)."""
        # In plain mode, just print some newlines for visual separation
        print("\n" * 3)

    # === PROGRESS COMPONENTS ===

    def create_spinner(self, message: str = "Loading...") -> PlainSpinner:
        """Create a plain text spinner."""
        return PlainSpinner(message)

    def create_progress(self, total: int = 100, description: str = "") -> PlainProgressBar:
        """Create a plain text progress bar."""
        return PlainProgressBar(total, description)

    # === INTERACTIVE COMPONENTS ===

    def select(
        self,
        choices: list[str],
        title: str = "Select an option",
        default_index: int = 0,
    ) -> str | None:
        """Show numbered selection menu."""
        print(f"\n{title}")
        for i, choice in enumerate(choices, 1):
            marker = "*" if i - 1 == default_index else " "
            print(f"  {marker}{i}. {choice}")
        print()

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
                print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
            except (EOFError, KeyboardInterrupt):
                return None

    def confirm(self, question: str, default: bool = True) -> bool | None:
        """Show yes/no prompt."""
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
        """Get text input from user."""
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt

    # === LIVE UPDATES ===

    def create_live_context(self) -> PlainLiveContext:
        """Create a plain text live context."""
        return PlainLiveContext()
