"""
Checklist management module for RagOps Agent CE.

Handles checklist file operations, formatting, and watching functionality.
Follows Single Responsibility Principle - manages only checklist-related operations.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


def _list_checklists() -> list[tuple[str, float]]:
    """Return list of all checklist files with their modification times."""
    checklist_dir = Path("ragops_checklists")
    if not checklist_dir.exists():
        return []

    checklists: list[tuple[str, float]] = []
    for file_path in checklist_dir.glob("checklist_*.json"):
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

    # Separator
    lines.append("")
    lines.append("[dim]" + "─" * 50 + "[/dim]")

    return "\n".join(lines)


class ChecklistWatcher:
    """
    Watches for checklist file changes and updates transcript in real-time.

    Manages background thread for file monitoring and transcript integration.
    """

    def __init__(self, transcript_ref: list[str], session_start_mtime: float | None = None):
        self.transcript_ref = transcript_ref
        self.session_start_mtime = session_start_mtime

        # Initialize last_seen_mtime to current newest file time to ignore existing checklists
        if session_start_mtime is not None:
            baseline = session_start_mtime
        else:
            _, latest_mtime = _latest_checklist()
            baseline = latest_mtime
        self.last_seen_mtime_ref = [baseline or 0.0]  # Mutable reference

        self.running = False
        self.thread: threading.Thread | None = None
        self._checklist_message_index: int | None = None
        self._last_update_time = 0.0

    def start(self) -> None:
        """Start the checklist watcher background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the checklist watcher background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _update_checklist_in_transcript(self, checklist_lines: str) -> None:
        """
        Previously injected checklist into transcript. Now we render checklist in a side panel,
        so avoid injecting into transcript to prevent duplication. Also remove any previously
        injected checklist lines if present.

        Args:
            checklist_lines: Formatted checklist content
        """
        # Remove previously injected checklist from transcript if exists
        try:
            if self._checklist_message_index is not None:
                idx = self._checklist_message_index
                if 0 <= idx < len(self.transcript_ref):
                    # Remove the checklist content line
                    self.transcript_ref.pop(idx)
                    # Try to remove header line if it exists right before
                    if (
                        idx - 1 >= 0
                        and self.transcript_ref[idx - 1].strip()
                        == "[dim]--- Checklist Created ---[/dim]"
                    ):
                        self.transcript_ref.pop(idx - 1)
                self._checklist_message_index = None
        except Exception:
            # Best-effort cleanup; ignore any errors
            self._checklist_message_index = None

        # No further action: checklist is rendered in the side panel now
        return

    def _watch_loop(self) -> None:
        """
        Background thread loop for monitoring checklist file changes.

        Polls for file changes and triggers transcript updates when needed.
        """
        while self.running:
            try:
                cur_name, cur_mtime = _latest_checklist()
                if cur_mtime and cur_mtime > self.last_seen_mtime_ref[0]:
                    if self.session_start_mtime and cur_mtime < self.session_start_mtime:
                        # Ignore checklists created before this session
                        time.sleep(0.1)
                        continue
                    current_checklist_data = _load_checklist(cur_name) if cur_name else None
                    new_checklist_content = format_checklist_compact(current_checklist_data)
                    self.last_seen_mtime_ref[0] = cur_mtime
                    self._last_update_time = time.time()
                    self._update_checklist_in_transcript(new_checklist_content)

                    # Notify that checklist was updated (callback can be added if needed)
                    self._on_checklist_updated()

                time.sleep(0.1)  # Check every 100ms
            except Exception:
                time.sleep(1.0)  # Longer sleep on error

    def _on_checklist_updated(self) -> None:
        """Hook for checklist update notifications. Override in subclass if needed."""
        pass


class ChecklistWatcherWithRenderer(ChecklistWatcher):
    """
    Extended ChecklistWatcher that integrates with ScreenRenderer.

    Automatically triggers screen re-rendering when checklist updates.
    """

    def __init__(
        self,
        transcript_ref: list[str],
        renderer=None,
        session_start_mtime: float | None = None,
    ):
        super().__init__(transcript_ref, session_start_mtime=session_start_mtime)
        self.renderer = renderer

    def _on_checklist_updated(self) -> None:
        """Trigger screen re-render when checklist is updated."""
        if self.renderer:
            try:
                cl_text = get_active_checklist_text(self.session_start_mtime)
                if cl_text:
                    self.renderer.render_conversation_and_checklist(
                        self.transcript_ref, cl_text, show_input_space=True
                    )
                else:
                    self.renderer.render_conversation_screen(
                        self.transcript_ref, show_input_space=True
                    )
            except Exception:
                pass  # Ignore render errors during input


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
    checklists = _list_checklists()
    if not checklists:
        return None

    for filename, mtime in reversed(checklists):
        if since_ts is not None and mtime < since_ts:
            continue
        data = _load_checklist(filename)
        if not data or "items" not in data:
            continue
        items = data.get("items", [])
        has_active = any(item.get("status", "pending") != "completed" for item in items)
        if not has_active:
            continue
        return format_checklist_compact(data)

    return None
