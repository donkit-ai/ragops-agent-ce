from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any
from typing import Callable

from ..db import kv_get
from ..db import migrate
from ..db import open_db
from ..interactive_input import interactive_confirm
from ..interactive_input import interactive_select
from ..llm import ToolFunction
from ..llm import ToolSpec


class AgentTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], str],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_tool_spec(self) -> ToolSpec:
        return ToolSpec(
            function=ToolFunction(
                name=self.name, description=self.description, parameters=self.parameters
            )
        )


# Built-in tools


def tool_time_now() -> AgentTool:
    def _handler(_: dict[str, Any]) -> str:
        now = _dt.datetime.now().isoformat()
        return now

    return AgentTool(
        name="time_now",
        description="Return current local datetime in ISO format",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_handler,
    )


def tool_db_get() -> AgentTool:
    def _handler(args: dict[str, Any]) -> str:
        key = str(args.get("key", ""))
        if not key:
            return ""
        with open_db() as db:
            migrate(db)
            val = kv_get(db, key)
            return "" if val is None else val

    return AgentTool(
        name="db_get",
        description="Get a value from local key-value store by key",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def tool_list_directory() -> AgentTool:
    def _handler(args: dict[str, Any]) -> str:
        path_str = str(args.get("path", "."))
        try:
            path = Path(path_str).expanduser().resolve()
            if not path.exists():
                return json.dumps({"error": f"Path does not exist: {path_str}"})
            if not path.is_dir():
                return json.dumps({"error": f"Path is not a directory: {path_str}"})
            items = []
            for item in sorted(path.iterdir()):
                try:
                    is_dir = item.is_dir()
                    size = None if is_dir else item.stat().st_size
                    items.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "is_directory": is_dir,
                            "size_bytes": size,
                        }
                    )
                except (PermissionError, OSError):
                    # Skip items we can't access
                    continue
            return json.dumps(
                {
                    "path": str(path),
                    "items": items,
                    "total_items": len(items),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    return AgentTool(
        name="list_directory",
        description=(
            "List contents of a directory with file/folder info. "
            "Returns JSON with items array containing name, path, is_directory, "
            "and size_bytes for each item."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list (supports ~ for home directory)",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def tool_read_file() -> AgentTool:
    def _handler(args: dict[str, Any]) -> str:
        file_path = args.get("path", "")
        offset = args.get("offset", 1)
        limit = args.get("limit", 100)

        if not file_path:
            return json.dumps({"error": "File path is required."})

        try:
            path_obj = Path(file_path).expanduser().resolve()

            if not path_obj.exists():
                return json.dumps({"error": f"File does not exist: {file_path}"})

            if not path_obj.is_file():
                return json.dumps({"error": f"Path is not a file: {file_path}"})

            # Read file content
            with open(path_obj, encoding="utf-8") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Validate offset and limit
            if offset < 1:
                offset = 1
            if limit < 1:
                limit = 100

            # Calculate range
            start_idx = offset - 1
            end_idx = min(start_idx + limit, total_lines)

            # Get requested lines
            selected_lines = lines[start_idx:end_idx]

            # Format output with line numbers
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=offset):
                formatted_lines.append(f"{i:6d}\t{line.rstrip()}")

            result = {
                "path": str(path_obj),
                "total_lines": total_lines,
                "showing_lines": f"{offset}-{end_idx}",
                "content": "\n".join(formatted_lines),
            }

            if end_idx < total_lines:
                result["note"] = f"File has more lines. Use offset={end_idx + 1} to continue."

            return json.dumps(result, ensure_ascii=False)

        except UnicodeDecodeError:
            return json.dumps({"error": "File is not a text file or has unsupported encoding."})
        except PermissionError:
            return json.dumps({"error": f"Permission denied: {file_path}"})
        except Exception as e:
            return json.dumps({"error": f"Failed to read file: {str(e)}"})

    return AgentTool(
        name="read_file",
        description=(
            "Reads and returns the content of a text file with line numbers. "
            "Supports pagination with offset and limit parameters for large files. "
            "Use this to examine file contents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (supports ~ for home directory).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed). Default: 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Default: 100.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def tool_grep() -> AgentTool:
    def _handler(args: dict[str, Any]) -> str:
        pattern = args.get("pattern", "")
        include = args.get("include", "")
        path = args.get("path", ".")

        if not pattern:
            return json.dumps({"error": "Pattern is required for grep."})

        # Compile regex pattern for filename search
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return json.dumps({"error": f"Invalid regex pattern: {e}"})

        # Resolve search path
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            return json.dumps({"error": f"Path does not exist: {path}"})

        # Prepare glob pattern for file filtering
        glob_pattern = include if include else "**/*"

        matches = []
        try:
            if path_obj.is_file():
                # Single file - check if name matches
                if regex.search(path_obj.name):
                    matches.append(
                        {
                            "type": "match",
                            "data": {
                                "path": {"text": str(path_obj)},
                                "name": path_obj.name,
                            },
                        }
                    )
            else:
                # Search recursively
                all_items = list(path_obj.rglob(glob_pattern.lstrip("*")))

                for item_path in all_items:
                    # Search in filename (not content)
                    if regex.search(item_path.name):
                        matches.append(
                            {
                                "type": "match",
                                "data": {
                                    "path": {"text": str(item_path)},
                                    "name": item_path.name,
                                    "is_directory": item_path.is_dir(),
                                },
                            }
                        )
                        # Limit to prevent huge outputs
                        if len(matches) >= 500:
                            matches.append(
                                {
                                    "type": "summary",
                                    "data": {"message": "Reached 500 match limit"},
                                }
                            )
                            return "\n".join(json.dumps(m) for m in matches)

            if not matches:
                matches.append({"type": "summary", "data": {"message": "No matches found"}})

            return "\n".join(json.dumps(m) for m in matches)
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    return AgentTool(
        name="grep",
        description=(
            "Searches for files by their names using regular expressions (case-insensitive). "
            "Returns JSON output of matching files with their paths. "
            "Does NOT search file contents, only filenames."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for."},
                "include": {
                    "type": "string",
                    "description": (
                        "File pattern to include in the search (e.g., '*.py', '*.{ts,tsx}')."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in. Defaults to current working directory."
                    ),
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def tool_interactive_user_choice() -> AgentTool:
    """Tool for interactive selection from multiple options using arrow keys."""
    def _handler(args: dict[str, Any]) -> str:
        title = str(args.get("title", "Select an option"))
        choices_list = args.get("choices", [])
        recommended_index = args.get("recommended_index")

        if not isinstance(choices_list, list) or len(choices_list) < 1:
            return json.dumps({"error": "choices must be a non-empty list of strings"})

        # Validate choices are strings
        choices = []
        for i, choice in enumerate(choices_list):
            if isinstance(choice, str):
                choices.append(choice)
            else:
                return json.dumps({"error": f"choice at index {i} must be a string"})

        # Enhance title with recommended option hint if provided
        enhanced_title = title
        if recommended_index is not None and 0 <= recommended_index < len(choices):
            enhanced_title = f"{title} (Recommended: {choices[recommended_index]})"

        # Call interactive selection
        selected = interactive_select(choices=choices, title=enhanced_title)

        if selected is None:
            return json.dumps({"cancelled": True, "selected_choice": None, "selected_index": None})

        # Find index of selected choice
        selected_index = choices.index(selected) if selected in choices else None

        return json.dumps(
            {
                "cancelled": False,
                "selected_choice": selected,
                "selected_index": selected_index,
            },
            ensure_ascii=False,
        )

    return AgentTool(
        name="interactive_user_choice",
        description=(
            "Present an interactive selection menu to the user with arrow key navigation. "
            "Use this tool ALWAYS when you need the user to choose from 2 or more options, "
            "including: configuration choices, numbered options, lists, or any multiple choice scenario. "
            "Even if you already described options in your text message, you MUST still call this tool. "
            "Do NOT ask 'Which option would you like?' in text - use this tool instead. "
            "Returns the selected choice and its index. "
            "The user can navigate with ↑/↓ arrow keys and select with Enter. "
            "Examples: chunk size options, vector DB selection, model choices, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title/description for the selection menu",
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of option strings to choose from (must have at least 2 options)",
                    "minItems": 2,
                },
                "recommended_index": {
                    "type": "integer",
                    "description": (
                        "Optional 0-based index of the recommended option "
                        "(will be highlighted, but user can still choose any option)"
                    ),
                },
            },
            "required": ["title", "choices"],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def tool_interactive_user_confirm() -> AgentTool:
    """Tool for interactive yes/no confirmation using arrow keys."""
    def _handler(args: dict[str, Any]) -> str:
        question = str(args.get("question", "Continue?"))
        default = args.get("default", True)

        if not isinstance(default, bool):
            default = True

        # Call interactive confirmation
        confirmed = interactive_confirm(question=question, default=default)

        if confirmed is None:
            return json.dumps({"cancelled": True, "confirmed": None})

        return json.dumps({"cancelled": False, "confirmed": confirmed}, ensure_ascii=False)

    return AgentTool(
        name="interactive_user_confirm",
        description=(
            "Present an interactive yes/no confirmation dialog to the user with arrow key navigation. "
            "Use this tool when you need to ask the user for confirmation (e.g., 'Continue?', 'Proceed?'). "
            "The user can navigate with ←/→ arrow keys and select with Enter, or press 'y'/'n' for quick selection. "
            "IMPORTANT: If confirmed=false or cancelled, STOP and wait for the user's next message. "
            "If confirmed=true, continue with the planned action."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user (e.g., 'Continue?', 'Proceed with this action?')",
                },
                "default": {
                    "type": "boolean",
                    "description": "Default value (true for Yes, false for No). Default: true",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
