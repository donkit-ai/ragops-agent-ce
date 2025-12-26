"""
Checklist management module for RAGOps Agent CE.

Handles checklist operations, formatting, and watching functionality.
Now uses database storage instead of file system.
Follows Single Responsibility Principle - manages only checklist-related operations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from donkit_ragops.db import close, kv_all_by_prefix, kv_get, open_db
from donkit_ragops.display import ScreenRenderer
from donkit_ragops.schemas.agent_schemas import AgentSettings


@dataclass
class ActiveChecklist:
    name: str | None = None


active_checklist = ActiveChecklist()


@dataclass
class ChecklistStatusLine:
    """Compact one-line checklist status for bottom toolbar display.

    Shows current task and progress: "⚡ Chunk documents (2/7)"
    """

    icon: str = ""
    description: str = ""
    completed: int = 0
    total: int = 0

    def format(self) -> str:
        """Format status as a compact string for toolbar."""
        if not self.total:
            return ""
        return f"{self.icon} {self.description} ({self.completed}/{self.total})"


class ChecklistStatusProvider:
    """Singleton provider for checklist status displayed in prompt toolbar.

    Thread-safe access to current checklist status for real-time updates.
    """

    _instance: ChecklistStatusProvider | None = None
    _status: ChecklistStatusLine

    def __new__(cls) -> ChecklistStatusProvider:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._status = ChecklistStatusLine()
        return cls._instance

    @property
    def status(self) -> ChecklistStatusLine:
        """Get current status line."""
        return self._status

    def update_from_checklist(self, checklist_data: dict[str, Any] | None) -> None:
        """Update status from checklist data.

        Args:
            checklist_data: Checklist dictionary with 'items' list
        """
        if not checklist_data or "items" not in checklist_data:
            self._status = ChecklistStatusLine()
            return

        items = checklist_data.get("items", [])
        if not items:
            self._status = ChecklistStatusLine()
            return

        # Count completed and find current item
        completed_count = 0
        current_item: dict[str, Any] | None = None

        for item in items:
            status = item.get("status", "pending")
            if status == "completed":
                completed_count += 1
            elif status == "in_progress" and current_item is None:
                current_item = item
            elif status == "pending" and current_item is None:
                current_item = item

        total = len(items)

        # Determine icon and description
        if completed_count == total:
            # All done
            self._status = ChecklistStatusLine(
                icon="✓",
                description="All tasks completed",
                completed=completed_count,
                total=total,
            )
        elif current_item:
            # In progress or next pending
            item_status = current_item.get("status", "pending")
            icon = "⚡" if item_status == "in_progress" else "○"
            desc = current_item.get("description", "")
            # Truncate long descriptions
            if len(desc) > 35:
                desc = desc[:32] + "..."
            self._status = ChecklistStatusLine(
                icon=icon,
                description=desc,
                completed=completed_count,
                total=total,
            )
        else:
            self._status = ChecklistStatusLine(
                icon="○",
                description="Checklist",
                completed=completed_count,
                total=total,
            )

    def clear(self) -> None:
        """Clear status (hide from toolbar)."""
        self._status = ChecklistStatusLine()

    def get_toolbar_tokens(self) -> list[tuple[str, str]]:
        """Get prompt_toolkit style tokens for bottom toolbar.

        Returns:
            List of (style, text) tuples for prompt_toolkit
        """
        if not self._status.total:
            return []

        icon = self._status.icon
        desc = self._status.description
        progress = f"({self._status.completed}/{self._status.total})"

        # Style based on status
        if icon == "✓":
            icon_style = "class:checklist.done"
        elif icon == "⚡":
            icon_style = "class:checklist.active"
        else:
            icon_style = "class:checklist.pending"

        return [
            ("class:checklist.separator", " │ "),
            (icon_style, f"{icon} "),
            ("class:checklist.description", desc),
            ("class:checklist.progress", f" {progress}"),
        ]


# Global singleton instance
checklist_status_provider = ChecklistStatusProvider()


def _list_checklists() -> list[tuple[str, dict[str, Any]]]:
    """Return list of all checklists from database with their data.

    Returns:
        list: List of (name, checklist_data) tuples
    """
    db = open_db()
    try:
        all_checklists = kv_all_by_prefix(db, "checklist_")
        result: list[tuple[str, dict[str, Any]]] = []
        for key, value in all_checklists:
            try:
                data = json.loads(value)
                # Extract name from key (remove "checklist_" prefix)
                name = key.replace("checklist_", "", 1)
                result.append((name, data))
            except json.JSONDecodeError:
                continue
        return result
    finally:
        close(db)


def _latest_checklist() -> tuple[str | None, dict[str, Any] | None]:
    """
    Find the most recent checklist.

    Returns:
        tuple: (name, data) or (None, None) if no checklists found
    """
    checklists = _list_checklists()
    if not checklists:
        return None, None
    # Return the last one (most recent)
    return checklists[-1]


def _load_checklist(name: str) -> dict[str, Any] | None:
    """
    Load checklist data from database.

    Args:
        name: Name of the checklist (without "checklist_" prefix)

    Returns:
        dict: Checklist data or None if loading fails
    """
    db = open_db()
    try:
        key = f"checklist_{name}"
        data_raw = kv_get(db, key)
        if data_raw is None:
            return None
        return json.loads(data_raw)
    except json.JSONDecodeError:
        return None
    finally:
        close(db)


def format_checklist_compact(checklist_data: dict[str, Any] | None) -> str:
    """
    Format checklist data into compact visual representation.

    Args:
        checklist_data: Checklist data dictionary

    Returns:
        str: Rich-formatted checklist string
    """
    if not checklist_data or "items" not in checklist_data:
        return "[dim]No checklist available[/dim]"

    lines = []

    # Header with bright styling
    lines.append("[white on blue] ✓ TODO [/white on blue]")
    lines.append("")

    # Items with status indicators
    for item in checklist_data["items"]:
        status = item.get("status", "pending")
        content = item.get("description", "")  # Use "description" field from JSON
        priority = item.get("priority", "medium")

        # Status icons with colors
        if status == "completed":
            icon = "[green]✓[/green]"
        elif status == "in_progress":
            icon = "[yellow]⚡[/yellow]"
        else:  # pending
            icon = "[dim]○[/dim]"

        # Priority styling
        if priority == "high":
            content_style = "[white]" + content + "[/white]"
        elif priority == "medium":
            content_style = content
        else:  # low
            content_style = "[dim]" + content + "[/dim]"

        lines.append(f"  {icon} {content_style}")

    return "\n".join(lines)


class _HistoryEntry(Protocol):
    """Protocol describing minimal interface of history entries used by helpers."""

    content: str | None


def _update_active_checklist_from_history(history: Sequence[_HistoryEntry]) -> None:
    """Update `active_checklist` name based on the latest tool response."""

    if not history:
        return
    try:
        tool_result = history[-1].content or "{}"
        parsed = json.loads(tool_result)
    except (AttributeError, json.JSONDecodeError, ValueError, TypeError, IndexError):
        return

    if isinstance(parsed, dict) and parsed.get("name"):
        # Store just the name without any extension
        active_checklist.name = parsed["name"]


def handle_checklist_tool_event(
    tool_name: str | None,
    history: Sequence[_HistoryEntry],
    *,
    renderer: ScreenRenderer | None,
    transcript: list[str],
    agent_settings: AgentSettings,
    session_start_mtime: float | None,
    render: bool,
) -> None:
    """Handle checklist-related tool events emitted by the agent stream."""

    # Updated tool names - now without "checklist_" prefix (built-in tools)
    if tool_name not in (
        "get_checklist",
        "create_checklist",
        "update_checklist_item",
    ):
        return

    _update_active_checklist_from_history(history)

    if not render or renderer is None:
        return

    try:
        cl_text = get_active_checklist_text(session_start_mtime)
        renderer.render_project(
            transcript,
            cl_text,
            agent_settings=agent_settings,
        )
    except Exception:
        pass


def get_current_checklist() -> str:
    """
    Get current checklist formatted for display.

    Returns:
        str: Rich-formatted checklist content
    """
    name, data = _latest_checklist()
    if not name or not data:
        return "[dim]No checklist found[/dim]"

    return format_checklist_compact(data)


def get_active_checklist_text(since_ts: float | None = None) -> str | None:
    """
    Return formatted checklist text only if there is at least one non-completed item.

    Args:
        since_ts: Only show checklists created after this timestamp (session start time)

    Returns:
        str | None: Rich-formatted checklist if active, otherwise None
    """

    def _has_active_items(data: dict[str, Any]) -> bool:
        """Check if checklist has any non-completed items."""
        if not data or "items" not in data:
            return False
        items = data.get("items", [])
        return any(item.get("status", "pending") != "completed" for item in items)

    checklists = _list_checklists()
    if not checklists:
        return None

    # Check active checklist first (explicitly loaded by user)
    # Don't filter by since_ts for explicitly activated checklists
    if active_checklist.name:
        data = _load_checklist(active_checklist.name)
        if data and _has_active_items(data):
            return format_checklist_compact(data)
        # Reset if no active items
        active_checklist.name = None

    # Find any checklist with active items created in this session
    for name, data in reversed(checklists):
        # Skip checklists created before session start
        if since_ts is not None and data.get("created_at", 0) < since_ts:
            continue
        if _has_active_items(data):
            return format_checklist_compact(data)

    return None
