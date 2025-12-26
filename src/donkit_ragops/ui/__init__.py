"""UI abstraction layer for RAGOps Agent CE.

Provides framework-agnostic UI interface that can be implemented
by different backends (Rich, PromptToolkit, plain text, etc.).

Usage:
    from donkit_ragops.ui import get_ui, StyleName

    ui = get_ui()
    ui.print("Hello!", StyleName.SUCCESS)
    ui.print_error("Something went wrong")

    with ui.create_spinner("Loading...") as spinner:
        # do work
        pass

    choice = ui.select(["Option 1", "Option 2"], "Choose:")

For enterprise mode (prompt_toolkit compatibility):
    from donkit_ragops.ui import set_ui_adapter, get_ui

    set_ui_adapter("prompt_toolkit")
    ui = get_ui()
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING

from donkit_ragops.ui.protocol import UI
from donkit_ragops.ui.styles import StyleName

if TYPE_CHECKING:
    pass

__all__ = ["UI", "StyleName", "UIAdapter", "get_ui", "set_ui_adapter", "reset_ui"]


class UIAdapter(str, Enum):
    """Available UI adapters."""

    RICH = "rich"
    PLAIN = "plain"
    PROMPT_TOOLKIT = "prompt_toolkit"


# Singleton UI instance
_ui_instance: UI | None = None
_ui_adapter: UIAdapter | None = UIAdapter.RICH


def set_ui_adapter(adapter: UIAdapter | str) -> None:
    """Set the UI adapter to use.

    Must be called before first get_ui() call.

    Args:
        adapter: UIAdapter enum or string name
    """
    global _ui_adapter, _ui_instance

    if isinstance(adapter, str):
        adapter = UIAdapter(adapter)

    _ui_adapter = adapter
    _ui_instance = None  # Reset instance to pick up new adapter


def get_ui(force_plain: bool = False) -> UI:
    """Get the UI instance.

    Auto-detects the best UI implementation unless set_ui_adapter() was called:
    - RichUI if running in a TTY
    - PlainUI if piped or force_plain=True
    - PromptToolkitUI if set_ui_adapter("prompt_toolkit") was called

    Args:
        force_plain: Force plain text output (for testing/pipes)

    Returns:
        UI implementation instance
    """
    global _ui_instance

    if _ui_instance is not None and not force_plain:
        return _ui_instance

    # Check if adapter was explicitly set
    if _ui_adapter == UIAdapter.PROMPT_TOOLKIT:
        from donkit_ragops.ui.adapters.prompt_toolkit_adapter import PromptToolkitUI

        _ui_instance = PromptToolkitUI()
    elif _ui_adapter == UIAdapter.PLAIN or force_plain or not sys.stdout.isatty():
        from donkit_ragops.ui.adapters.plain_adapter import PlainUI

        _ui_instance = PlainUI()
    else:
        from donkit_ragops.ui.adapters.rich_adapter import RichUI

        _ui_instance = RichUI()

    return _ui_instance


def reset_ui() -> None:
    """Reset the UI instance (for testing)."""
    global _ui_instance, _ui_adapter
    _ui_instance = None
    _ui_adapter = None
