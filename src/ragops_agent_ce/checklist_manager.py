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

from ragops_agent_ce.db import close, kv_all_by_prefix, kv_get, open_db
from ragops_agent_ce.display import ScreenRenderer
from ragops_agent_ce.schemas.agent_schemas import AgentSettings


@dataclass
class ActiveChecklist:
    name: str | None = None


active_checklist = ActiveChecklist()


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
