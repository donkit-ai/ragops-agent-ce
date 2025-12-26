"""Semantic style definitions for UI abstraction.

StyleName enum provides framework-agnostic style names that can be
mapped to specific framework styles (Rich markup, ANSI codes, etc.).
"""

from __future__ import annotations

from enum import Enum


class StyleName(Enum):
    """Semantic styles - framework agnostic.

    These represent the meaning/purpose of the style, not the visual appearance.
    Each UI adapter maps these to framework-specific styles.
    """

    # Message types
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"

    # Text modifiers
    DIM = "dim"
    BOLD = "bold"
    ITALIC = "italic"

    # UI elements
    AGENT_PREFIX = "agent_prefix"
    USER_PREFIX = "user_prefix"
    TOOL_NAME = "tool_name"
    TOOL_ARGS = "tool_args"

    # Panel/border styles
    PANEL_BORDER = "panel_border"
    PANEL_TITLE = "panel_title"

    # Selection/highlight
    SELECTED = "selected"
    HIGHLIGHTED = "highlighted"
    CURSOR = "cursor"

    # Code/technical
    CODE = "code"
    PATH = "path"
    URL = "url"

    # Progress/status
    PROGRESS = "progress"
    SPINNER = "spinner"

    # Keyboard hints
    KEY = "key"
    KEY_HINT = "key_hint"


# Rich markup mapping
RICH_STYLES: dict[StyleName, str] = {
    # Message types
    StyleName.ERROR: "bold red",
    StyleName.SUCCESS: "bold green",
    StyleName.WARNING: "bold yellow",
    StyleName.INFO: "bold blue",
    # Text modifiers
    StyleName.DIM: "dim",
    StyleName.BOLD: "bold",
    StyleName.ITALIC: "italic",
    # UI elements
    StyleName.AGENT_PREFIX: "bold green",
    StyleName.USER_PREFIX: "bold",
    StyleName.TOOL_NAME: "yellow",
    StyleName.TOOL_ARGS: "dim",
    # Panel/border
    StyleName.PANEL_BORDER: "cyan",
    StyleName.PANEL_TITLE: "bold cyan",
    # Selection
    StyleName.SELECTED: "bold on grey11",
    StyleName.HIGHLIGHTED: "bold cyan",
    StyleName.CURSOR: "bold cyan",
    # Code/technical
    StyleName.CODE: "cyan",
    StyleName.PATH: "blue underline",
    StyleName.URL: "blue underline",
    # Progress
    StyleName.PROGRESS: "cyan",
    StyleName.SPINNER: "cyan",
    # Keys
    StyleName.KEY: "bold yellow",
    StyleName.KEY_HINT: "dim",
}


def get_rich_style(style: StyleName) -> str:
    """Get Rich markup style string for a semantic style.

    Args:
        style: Semantic style name

    Returns:
        Rich markup style string
    """
    return RICH_STYLES.get(style, "")


# StyledText type for semantic styling
StyledSegment = tuple[StyleName | None, str]
StyledText = list[StyledSegment]


def styled_text(*segments: StyledSegment) -> StyledText:
    """Create a StyledText from segments.

    Args:
        *segments: Tuples of (style, text)

    Returns:
        StyledText list

    Example:
        styled_text(
            (StyleName.DIM, "Loading: "),
            (StyleName.BOLD, "file.txt"),
        )
    """
    return list(segments)
