"""
Checklist management module for RAGOps Agent CE.

Handles checklist file operations, formatting, and watching functionality.
Follows Single Responsibility Principle - manages only checklist-related operations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Protocol

from ragops_agent_ce.display import ScreenRenderer
from ragops_agent_ce.schemas.agent_schemas import AgentSettings


@dataclass
class ActiveChecklist:
    name: str | None = None


active_checklist = ActiveChecklist()


def _list_checklists() -> list[tuple[str, float]]:
    """Return list of all checklist files with their modification times."""
    checklist_dir = Path("ragops_checklists")
    if not checklist_dir.exists():
        return []

    checklists: list[tuple[str, float]] = []
    for file_path in checklist_dir.glob("*.json"):
        try:
            mtime = file_path.stat().st_mtime
            checklists.append((str(file_path.name), mtime))
        except OSError:
            continue

    return sorted(checklists, key=lambda item: item[1])


def _latest_checklist() -> tuple[str | None, float | None]:
    """
    Find the most recent checklist file.

    Returns:
        tuple: (filename, mtime) or (None, None) if no checklists found
    """
    checklists = _list_checklists()
    if not checklists:
        return None, None
    return checklists[-1]


def _load_checklist(filename: str) -> dict[str, Any] | None:
    """
    Load checklist data from JSON file.

    Args:
        filename: Name of the checklist file

    Returns:
        dict: Checklist data or None if loading fails
    """
    try:
        file_path = Path("ragops_checklists") / filename
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


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
        active_checklist.name = f"{parsed['name']}.json"


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

    if tool_name not in (
        "checklist_get_checklist",
        "checklist_create_checklist",
        "checklist_update_checklist_item",
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
    filename, _ = _latest_checklist()
    if not filename:
        return "[dim]No checklist found[/dim]"

    checklist_data = _load_checklist(filename)
    return format_checklist_compact(checklist_data)


def get_active_checklist_text(since_ts: float | None = None) -> str | None:
    """
    Return formatted checklist text only if there is at least one non-completed item.

    Returns:
        str | None: Rich-formatted checklist if active, otherwise None
    """

    def _get_checklist(filename: str) -> str | None:
        data = _load_checklist(filename)
        if not data or "items" not in data:
            return None
        items = data.get("items", [])
        has_active = any(item.get("status", "pending") != "completed" for item in items)
        if not has_active:
            return None
        return format_checklist_compact(data)

    checklists = _list_checklists()
    if not checklists:
        return None

    if active_checklist.name:
        checklist = _get_checklist(active_checklist.name)
        if checklist is None:
            active_checklist.name = None
        else:
            return checklist

    for filename, mtime in reversed(checklists):
        if since_ts is not None and mtime < since_ts:
            continue
        data = _get_checklist(filename)
        if data is not None:
            return data

    return None
