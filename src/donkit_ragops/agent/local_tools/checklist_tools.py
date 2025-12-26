from __future__ import annotations

import json
import time
from typing import Any, Literal

from donkit_ragops.checklist_manager import checklist_status_provider
from donkit_ragops.db import close, kv_get, kv_set, open_db

from .tools import AgentTool


def _checklist_key(name: str) -> str:
    """Generate database key for checklist."""
    return f"checklist_{name}"


def _save_checklist_to_db(name: str, checklist_data: dict[str, Any]) -> None:
    """Save checklist to database."""
    db = open_db()
    try:
        key = _checklist_key(name)
        kv_set(db, key, json.dumps(checklist_data))
    finally:
        close(db)


def _load_checklist_from_db(name: str) -> dict[str, Any] | None:
    """Load checklist from database."""
    db = open_db()
    try:
        key = _checklist_key(name)
        data_raw = kv_get(db, key)
        if data_raw is None:
            return None
        return json.loads(data_raw)
    finally:
        close(db)


def tool_create_checklist() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        name = payload.get("name")
        items = payload.get("items")

        if not name:
            return "Error: name is required."
        if not items or not isinstance(items, list):
            return "Error: items must be a non-empty list of task descriptions."

        # Check if checklist already exists
        existing_checklist = _load_checklist_from_db(name)
        if existing_checklist is not None:
            # Update status line to show existing checklist
            checklist_status_provider.update_from_checklist(existing_checklist)
            return (
                f"Checklist '{name}' already exists. "
                f"Returning existing checklist with {len(existing_checklist['items'])} items.\n\n"
                + json.dumps(existing_checklist, indent=2)
            )

        # Create new checklist
        checklist_items = [
            {
                "id": f"item_{i}",
                "description": item,
                "status": "pending",
            }
            for i, item in enumerate(items)
        ]

        checklist_data = {
            "name": name,
            "items": checklist_items,
            "created_at": time.time(),
        }

        _save_checklist_to_db(name, checklist_data)

        # Update status line for toolbar display
        checklist_status_provider.update_from_checklist(checklist_data)

        return f"Checklist '{name}' created with {len(items)} items.\n\n" + json.dumps(
            checklist_data, indent=2
        )

    return AgentTool(
        name="create_checklist",
        description=(
            "Creates a new checklist with a given name and list of tasks. "
            "If checklist already exists, returns the existing one. "
            "Each item gets an auto-generated ID (item_0, item_1, etc.) "
            "and starts with 'pending' status."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique checklist name (used as identifier).",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of task descriptions. "
                        "IDs will be auto-generated as 'item_<index>'."
                    ),
                },
            },
            "required": ["name", "items"],
        },
        handler=handler,
    )


def tool_get_checklist() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        name = payload.get("name")
        if not name:
            return "Error: name is required."

        checklist = _load_checklist_from_db(name)
        if checklist is None:
            return f"Checklist '{name}' not found."

        # Update status line to show this checklist
        checklist_status_provider.update_from_checklist(checklist)

        return json.dumps(checklist, indent=2)

    return AgentTool(
        name="get_checklist",
        description="Retrieves the current state of a checklist by its name as a JSON string.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Checklist name to load.",
                },
            },
            "required": ["name"],
        },
        handler=handler,
    )


def tool_update_checklist_item() -> AgentTool:
    def handler(payload: dict[str, Any]) -> str:
        name = payload.get("name")
        item_id = payload.get("item_id")
        status = payload.get("status")

        if not name:
            return "Error: name is required."
        if not item_id:
            return "Error: item_id is required."
        if not status:
            return "Error: status is required."

        # Validate status
        valid_statuses: list[Literal["pending", "in_progress", "completed", "failed"]] = [
            "pending",
            "in_progress",
            "completed",
            "failed",
        ]
        if status not in valid_statuses:
            return f"Error: status must be one of {valid_statuses}."

        checklist = _load_checklist_from_db(name)
        if checklist is None:
            return f"Checklist '{name}' not found."

        # Find item index
        item_index = None
        items = checklist.get("items", [])
        for idx, item in enumerate(items):
            if item.get("id") == item_id:
                item_index = idx
                break

        if item_index is None:
            return f"Item '{item_id}' not found in checklist '{name}'."

        # Validate sequential execution: can't start new item if previous ones aren't completed
        if status == "in_progress" and item_index > 0:
            for prev_idx in range(item_index):
                prev_item = items[prev_idx]
                if prev_item.get("status") != "completed":
                    return (
                        f"Cannot start item '{item_id}': "
                        f"Previous item '{prev_item.get('id')}' ({prev_item.get('description')}) "
                        f"is not completed (status: {prev_item.get('status')}). "
                        f"Please complete previous items first."
                    )

        # Update status
        items[item_index]["status"] = status
        checklist["items"] = items

        _save_checklist_to_db(name, checklist)

        # Update status line for toolbar display
        checklist_status_provider.update_from_checklist(checklist)

        return f"Updated item '{item_id}' in checklist '{name}' to status '{status}'."

    return AgentTool(
        name="update_checklist_item",
        description=(
            "Updates the status of a specific item in a checklist. "
            "Enforces sequential execution: cannot set item to 'in_progress' "
            "if previous items are not 'completed'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Checklist name that contains the item.",
                },
                "item_id": {
                    "type": "string",
                    "description": "ID of the item to update (e.g., 'item_0').",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "failed"],
                    "description": "New status value to set for the item.",
                },
            },
            "required": ["name", "item_id", "status"],
        },
        handler=handler,
    )
